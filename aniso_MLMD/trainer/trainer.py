import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import torch.nn as nn
import wandb

from aniso_MLMD.trainer.data_loader import AnisoDataLoader, load_datasets, \
    load_overfit_data
from aniso_MLMD.model import ForTorPredictorNN, EnergyPredictor


class MLTrainer:
    def __init__(self, config, job_id, resume=False):
        print("***************CUDA: ", torch.cuda.is_available())
        self.resume = resume
        self.job_id = job_id
        self.project = config.project
        self.group = config.group
        self.notes = config.notes
        self.tags = config.tags
        self.log = config.log
        self.target = config.target

        # dataset parameters
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.shrink = config.shrink
        self.overfit = config.overfit
        self.processor_type = config.processor_type
        self.scale_range = config.scale_range

        # model parameters
        self.in_dim = config.in_dim
        self.neighbor_hidden_dim = config.neighbor_hidden_dim
        self.particle_hidden_dim = config.particle_hidden_dim
        self.n_layer = config.n_layer
        self.act_fn = config.act_fn
        self.dropout = config.dropout
        self.batch_norm = config.batch_norm
        self.neighbor_pool = config.neighbor_pool
        self.particle_pool = config.particle_pool
        self.box_len = config.box_len

        # optimizer parameters
        self.optim = config.optim
        self.lr = config.lr
        self.min_lr = config.min_lr
        self.decay = config.decay
        self.use_scheduler = config.use_scheduler
        self.scheduler_type = config.scheduler_type
        self.scheduler_patience = config.scheduler_patience
        self.scheduler_threshold = config.scheduler_threshold
        self.loss_type = config.loss_type

        # prior energy parameters
        self.prior_energy = config.prior_energy
        self.prior_energy_sigma = None
        self.prior_energy_n = None
        if config.prior_energy:
            self.prior_energy_sigma = config.prior_energy_sigma
            self.prior_energy_n = config.prior_energy_n

        self.prior_force = config.prior_force
        self.prior_force_sigma = None
        self.prior_force_n = None
        if config.prior_force:
            self.prior_force_sigma = config.prior_force_sigma
            self.prior_force_n = config.prior_force_n

        # run parameters
        self.epochs = config.epochs

        # select device (GPU or CPU)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # create data loaders
        self.aniso_data_loader = AnisoDataLoader(config.data_path,
                                                 config.batch_size,
                                                 config.shrink,
                                                 config.overfit,
                                                 processor_type=config.processor_type,
                                                 scale_range=config.scale_range)

        self.train_dataloader = self.aniso_data_loader.get_train_dataset()
        if not self.overfit:
            self.valid_dataloader = self.aniso_data_loader.get_valid_dataset()
            self.test_dataloader = self.aniso_data_loader.get_test_dataset()

        self.train_size = self.train_dataloader.dataset.__len__()
        self.n_batch = len(self.train_dataloader)
        self.valid_size = None
        self.test_size = None
        if not self.overfit:
            self.valid_size = self.valid_dataloader.dataset.__len__()
            self.test_size = self.test_dataloader.dataset.__len__()

        # create model
        self.model = self._create_model()

        print(self.model)

        # create loss, optimizer and schedule
        if self.loss_type == "mae":
            if self.target == "energy":
                self.force_loss = nn.L1Loss().to(self.device)
                self.torque_loss = nn.L1Loss().to(self.device)
            else:
                self.target_loss = nn.L1Loss().to(self.device)
            self.criteria = nn.L1Loss().to(self.device)
        elif self.loss_type == "mse":
            if self.target == "energy":
                self.force_loss = nn.MSELoss().to(self.device)
                self.torque_loss = nn.MSELoss().to(self.device)
            else:
                self.target_loss = nn.MSELoss().to(self.device)
            self.criteria = nn.MSELoss().to(self.device)
        elif self.loss_type == "huber":
            if self.target == "energy":
                self.force_loss = nn.HuberLoss(delta=5).to(self.device)
                self.torque_loss = nn.HuberLoss(delta=5).to(self.device)
            else:
                self.target_loss = nn.HuberLoss(delta=5).to(self.device)
            self.criteria = nn.HuberLoss(delta=5).to(self.device)

        if self.optim == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.lr,
                                              weight_decay=self.decay)
        if self.optim == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.decay)

        self.set_scheduler()

        if resume:
            self.load_last_state()

        self.wandb_config = self._create_config()
        self._initiate_wandb_run()

    def set_scheduler(self):
        if self.scheduler_type == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.98, patience=self.scheduler_patience,
                min_lr=self.min_lr, threshold=self.scheduler_threshold,
                verbose=True, cooldown=30)
        elif self.scheduler_type == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=5000,
                                                             gamma=0.98,
                                                             verbose=True)

    def _create_model(self):
        if self.target == "energy":

            model = EnergyPredictor(in_dim=self.in_dim,
                                    neighbor_hidden_dim=self.neighbor_hidden_dim,
                                    particle_hidden_dim=self.particle_hidden_dim,
                                    box_len=self.box_len,
                                    n_layers=self.n_layer,
                                    act_fn=self.act_fn,
                                    dropout=self.dropout,
                                    batch_norm=self.batch_norm,
                                    device=self.device,
                                    neighbor_pool=self.neighbor_pool,
                                    particle_pool=self.particle_pool,
                                    prior_energy=self.prior_energy,
                                    prior_energy_sigma=self.prior_energy_sigma,
                                    prior_energy_n=self.prior_energy_n
                                    )
        else:
            model = ForTorPredictorNN(in_dim=self.in_dim,
                                      neighbor_hidden_dim=self.neighbor_hidden_dim,
                                      box_len=self.box_len,
                                      n_layers=self.n_layer,
                                      act_fn=self.act_fn,
                                      dropout=self.dropout,
                                      batch_norm=self.batch_norm,
                                      device=self.device,
                                      neighbor_pool=self.neighbor_pool,
                                      prior_force=self.prior_force,
                                      prior_force_sigma=self.prior_force_sigma,
                                      prior_force_n=self.prior_force_n
                                      )

        model.to(self.device)
        return model

    def load_last_state(self):
        print('reloading latest model and optimizer state...')
        last_checkpoint = torch.load('last_checkpoint.pth')
        self.model.load_state_dict(last_checkpoint['model'])
        self.optimizer.load_state_dict(last_checkpoint['optimizer'])
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] = 0.01
        print('successfully loaded model and optimizer...')

    def _create_config(self):
        config = {
            "target": self.target,
            "overfit": self.overfit,
            "batch_size": self.batch_size,
            "shrink": self.shrink,
            "lr": self.lr,
            "in_dim": self.in_dim,
            "neighbor_hidden_dim": self.neighbor_hidden_dim,
            "particle_hidden_dim": self.particle_hidden_dim,
            "n_layer": self.n_layer,
            "neighbor_pool": self.neighbor_pool,
            "particle_pool": self.particle_pool,
            "optim": self.optim,
            "decay": self.decay,
            "act_fn": self.act_fn,
            "dropout": self.dropout,
            "use_scheduler": self.use_scheduler,
            "scheduler_type": self.scheduler_type,
            "scheduler_patience": self.scheduler_patience,
            "scheduler_threshold": self.scheduler_threshold,
            "loss_type": self.loss_type,
            "batch_norm": self.batch_norm,
            "resume": self.resume,
            "prior_energy": self.prior_energy,
            "prior_energy_sigma": self.prior_energy_sigma,
            "prior_energy_n": self.prior_energy_n,
            "prior_force": self.prior_force,
            "prior_force_sigma": self.prior_force_sigma,
            "prior_force_n": self.prior_force_n,

        }
        print("***************** Config *******************")
        print(config)

        return config

    def _initiate_wandb_run(self):
        if self.log:
            self.wandb_config = self._create_config()

            self.wandb_run = wandb.init(project=self.project, notes=self.notes,
                                        group=self.group,
                                        tags=self.tags,
                                        config=self.wandb_config)
            self.wandb_run_name = self.wandb_run.name
            self.wandb_run_path = self.wandb_run.path
            self.wandb_run.summary["job_id"] = self.job_id
            self.wandb_run.summary["data_path"] = self.data_path
            self.wandb_run.summary["in_dim"] = self.in_dim
            self.wandb_run.summary["train_size"] = self.train_size
            self.wandb_run.summary["valid_size"] = self.valid_size
            self.wandb_run.summary["test_size"] = self.test_size

    def _calculate_torque(self, torque_grad, R1):

        tq_x = torch.cross(torque_grad[:, :, :, 0], R1[:, :, :, 0])
        tq_y = torch.cross(torque_grad[:, :, :, 1], R1[:, :, :, 1])
        tq_z = torch.cross(torque_grad[:, :, :, 2], R1[:, :, :, 2])
        predicted_torque = tq_x + tq_y + tq_z
        return predicted_torque.to(self.device)

    def clip_grad(self, model, max_norm):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
        total_norm = total_norm ** (0.5)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in model.parameters():
                p.grad.data.mul_(clip_coef)
        return total_norm

    def _train(self):
        self.model.train()
        train_loss = 0.
        running_loss = 0.
        force_running_loss = 0.
        torque_running_loss = 0.
        for i, (
                (positions, orientation_q, orientation_R,
                 neighbor_list), target_force,
                target_torque,
                energy) in enumerate(
            self.train_dataloader):
            self.optimizer.zero_grad()
            positions.requires_grad = True
            orientation_R.requires_grad = True
            positions = positions.to(self.device)
            orientation_R = orientation_R.to(self.device)

            _prediction = self.model(positions, orientation_R, neighbor_list)
            if self.target == "energy":
                energy_prediction = _prediction
                predicted_force = - torch.autograd.grad(energy_prediction.sum(),
                                                        positions,
                                                        create_graph=True)[
                    0].to(
                    self.device)
                torque_grad = - torch.autograd.grad(energy_prediction.sum(),
                                                    orientation_R,
                                                    create_graph=True)[0]
                predicted_torque = self._calculate_torque(torque_grad,
                                                          orientation_R)
                target_force = target_force.to(self.device)
                target_torque = target_torque.to(self.device)

                force_loss = self.force_loss(predicted_force, target_force)
                torque_loss = self.torque_loss(predicted_torque, target_torque)
                _loss = force_loss + torque_loss
                force_running_loss += force_loss.item()
                torque_running_loss += torque_loss.item()

            elif self.target == "force":
                target_val = target_force.to(self.device)
                _loss = self.target_loss(_prediction, target_val)
            else:
                target_val = target_torque.to(self.device)
                _loss = self.target_loss(_prediction, target_val)

            train_loss += _loss
            running_loss += _loss.item()

            _loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            if i % 100 == 99:
                if self.log:
                    if self.target == "energy":
                        wandb.log({'running_loss': running_loss / 99.,
                                   'force_running_loss': force_running_loss / 99.,
                                   'torque_running_loss': torque_running_loss / 99.,
                                   })
                    else:
                        wandb.log({'running_loss': running_loss / 99.})

                running_loss = 0.0

                force_running_loss = 0.0
                torque_running_loss = 0.0
            del positions, orientation_R, orientation_q, \
                _prediction, _loss

        train_loss = train_loss / (i + 1)

        return train_loss.item()

    def _validation(self, data_loader, print_output=False):
        self.model.eval()
        # with torch.no_grad():
        total_error = 0.
        total_force_error = 0.
        total_torque_error = 0.
        for i, (
                (positions, orientation_q, orientation_R, neighbor_list),
                target_force,
                target_torque,
                energy) in enumerate(
            data_loader):
            positions.requires_grad = True
            orientation_R.requires_grad = True
            positions = positions.to(self.device)
            orientation_R = orientation_R.to(self.device)

            _prediction = self.model(positions, orientation_R, neighbor_list)

            if self.target == "energy":
                energy_prediction = _prediction

                # force prediction
                predicted_force = - torch.autograd.grad(energy_prediction.sum(),
                                                        positions,
                                                        create_graph=True)[
                    0].to(
                    self.device)
                target_force = target_force.to(self.device)
                if self.aniso_data_loader.force_scaler is not None:
                    predicted_force = self.aniso_data_loader.force_scaler.inv_transform(
                        predicted_force)
                    target_force = self.aniso_data_loader.force_scaler.inv_transform(
                        target_force)

                force_error = self.criteria(predicted_force,
                                            target_force).item()

                # torque prediction
                torque_grad = - torch.autograd.grad(energy_prediction.sum(),
                                                    orientation_R,
                                                    create_graph=True)[0]

                predicted_torque = self._calculate_torque(torque_grad,
                                                          orientation_R)

                target_torque = target_torque.to(self.device)
                if self.aniso_data_loader.torque_scaler is not None:
                    predicted_torque = self.aniso_data_loader.torque_scaler.inv_transform(
                        predicted_torque)
                    target_torque = self.aniso_data_loader.torque_scaler.inv_transform(
                        target_torque)

                torque_error = self.criteria(predicted_torque,
                                             target_torque).item()

                _error = (force_error + torque_error)
            elif self.target == "force":
                target_val = target_force.to(self.device)
                if self.aniso_data_loader.force_scaler is not None:
                    _prediction = self.aniso_data_loader.force_scaler.inv_transform(
                        _prediction)
                    target_val = self.aniso_data_loader.force_scaler.inv_transform(
                        target_val)
                _error = self.criteria(_prediction, target_val).item()
            else:
                target_val = target_torque.to(self.device)
                if self.aniso_data_loader.torque_scaler is not None:
                    _prediction = self.aniso_data_loader.torque_scaler.inv_transform(
                        _prediction)
                    target_val = self.aniso_data_loader.torque_scaler.inv_transform(
                        target_val)
                _error = self.criteria(_prediction, target_val).item()

            
            total_error += _error


            if print_output and i % 100 == 0:
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                if self.target == "energy":
                    print("force prediction: ", predicted_force[5][:10])
                    print("force target: ", target_force[5][:10])
                    print("torque prediction: ", predicted_torque[5][:10])
                    print("torque target: ", target_torque[5][:10])
                else:
                    print("prediction: ", _prediction[5][:10])
                    print("target: ", target_val[5][:10])

                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

        return total_error / (i + 1), total_force_error / (
                i + 1), total_torque_error / (i + 1)

    def run(self):
        if self.overfit:
            self._run_overfit()
        else:
            self._run_train_valid()

    def _run_overfit(self):
        if self.log:
            wandb.watch(models=self.model, criterion=self.force_loss,
                        log="gradients",
                        log_freq=200)
        print(
            '**************************Overfitting*******************************')

        for epoch in range(self.epochs):
            train_loss = self._train()
            if self.use_scheduler:
                if self.scheduler_type == "ReduceLROnPlateau":
                    self.scheduler.step(train_loss)
                else:
                    self.scheduler.step()
            if epoch % 100 == 0:
                print('####### epoch {}/{}: \n\t train_loss: {}'.
                      format(epoch + 1, self.epochs, train_loss))

                # wandb.log({'train_loss': train_loss,
                #            "learning rate": self.optimizer.param_groups[0]['lr']})
        if self.log:
            wandb.finish()
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, 'last_checkpoint.pth')

    def _run_train_valid(self):
        if self.log:
            if self.target == "energy":
                wandb.watch(models=self.model, criterion=self.force_loss,
                            log="gradients",
                            log_freq=200)
            else:
                wandb.watch(models=self.model, criterion=self.target_loss,
                            log="gradients",
                            log_freq=200)
        print(
            '**************************Training*******************************')
        self.best_val_error = None

        for epoch in range(self.epochs):

            train_loss = self._train()

            # val_error = self._validation(self.valid_dataloader)
            # if self.use_scheduler:
            #     self.scheduler.step(val_error)
            if epoch % 10 == 0:
                val_error, val_force_error, val_torque_error = self._validation(
                    self.valid_dataloader, print_output=True)
                if self.use_scheduler:
                    self.scheduler.step(val_error)
                print('epoch {}/{}: \n\t train_loss: {}, \n\t val_error: {}'.
                      format(epoch + 1, self.epochs, train_loss, val_error))
                if self.log:
                    if self.target == "energy":
                        wandb.log({'train_loss': train_loss,
                                   'valid error': val_error,
                                   'valid force error': val_force_error,
                                   'valid torque error': val_torque_error,
                                   "learning rate":
                                       self.optimizer.param_groups[0][
                                           'lr']})
                    else:
                        wandb.log({'train_loss': train_loss,
                                   'valid error': val_error,
                                   "learning rate":
                                       self.optimizer.param_groups[0][
                                           'lr']})

                if self.best_val_error is None:
                    self.best_val_error = val_error

                if val_error <= self.best_val_error:
                    self.best_val_error = val_error
                    checkpoint = {
                        'epoch': epoch,
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    }
                    torch.save(checkpoint, 'best_model_checkpoint.pth')
                    print(
                        '#################################################################')
                    print('best_val_error: {}, best_epoch: {}'.format(
                        self.best_val_error, epoch))
                    print(
                        '#################################################################')
                    if self.log:
                        self.wandb_run.summary["best_epoch"] = epoch + 1
                        self.wandb_run.summary[
                            "best_val_error"] = self.best_val_error

        # Testing
        print(
            '**************************Testing*******************************')
        self.test_error, self.test_force_error, self.test_torque_error = self._validation(
            self.test_dataloader,
            print_output=True)
        print('Testing \n\t test error: {}'.
              format(self.test_error))
        if self.log:
            self.wandb_run.summary['test error'] = self.test_error
            wandb.finish()
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, 'last_checkpoint.pth')
