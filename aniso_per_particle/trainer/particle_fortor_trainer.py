import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import torch.nn as nn
import wandb

from aniso_per_particle.trainer.data_loader import AnisoParticleDataLoader
from aniso_per_particle.model import ParticleTorForPredictor_V1


class MSELoss(nn.Module):
    """Mean Squared Error Loss"""

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred_force, target_force, pred_torque, target_torque):
        return torch.mean((pred_force - target_force) ** 2) + torch.mean(
            (pred_torque - target_torque) ** 2)


class MAELoss(nn.Module):
    """Mean Absolute Error Loss"""

    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, pred_force, target_force, pred_torque, target_torque):
        return torch.mean(torch.abs(pred_force - target_force)) + torch.mean(
            torch.abs(pred_torque - target_torque))


class MSLLoss(nn.Module):
    """Mean Squared Logarithmic Error Loss"""

    def __init__(self):
        super(MSLLoss, self).__init__()

    def forward(self, pred_force, target_force, pred_torque, target_torque):
        return torch.mean((torch.log(pred_force + 1) - torch.log(
            target_force + 1)) ** 2) + torch.mean(
            (torch.log(pred_torque + 1) - torch.log(target_torque + 1)) ** 2)


class ForTorTrainer:
    def __init__(self, config, job_id, ):
        print("***************CUDA: ", torch.cuda.is_available())
        self.resume = config.resume
        self.job_id = job_id
        self.project = config.project
        self.group = config.group
        self.log = config.log

        # dataset parameters
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.overfit = config.overfit
        self.shrink = config.shrink
        self.train_idx = config.train_idx

        # model parameters
        self.in_dim = config.in_dim

        self.force_net_config = {
            "hidden_dim": config.force_hidden_dim,
            "n_layers": config.force_n_layers,
            "act_fn": config.force_act_fn,
            "pre_factor": config.force_pre_factor,
        }
        self.torque_net_config = {
            "hidden_dim": config.torque_hidden_dim,
            "n_layers": config.torque_n_layers,
            "act_fn": config.torque_act_fn,
            "pre_factor": config.torque_pre_factor,
        }
        self.ellipsoid_config = {'lpar': config.lpar,
                                 'lperp': config.lperp,
                                 'sigma': config.sigma,
                                 'n_factor': config.n_factor}

        self.dropout = config.dropout
        self.batch_norm = config.batch_norm
        self.initial_weight = config.initial_weight

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
        self.clipping = config.clipping

        # run parameters
        self.epochs = config.epochs

        # select device (GPU or CPU)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # create data loaders
        self.aniso_data_loader = AnisoParticleDataLoader(config.data_path,
                                                         batch_size=config.batch_size,
                                                         overfit=config.overfit,
                                                         shrink=config.shrink,
                                                         train_idx=self.train_idx)

        self.train_dataloader = self.aniso_data_loader.get_train_dataset()
        if not self.overfit:
            self.valid_dataloader = self.aniso_data_loader.get_valid_dataset()

        self.train_size = self.train_dataloader.dataset.__len__()
        self.n_batch = len(self.train_dataloader)
        self.valid_size = None
        if not self.overfit:
            self.valid_size = self.valid_dataloader.dataset.__len__()

        # create model
        self.model = self._create_model()

        print(self.model)

        # create loss, optimizer and schedule
        if self.loss_type == "mae":
            self.loss = MAELoss().to(self.device)
        elif self.loss_type == "mse":
            self.loss = MSELoss().to(self.device)
        elif self.loss_type == "msl":
            self.loss = MSLLoss().to(self.device)

        if self.optim == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.lr,
                                              weight_decay=self.decay)
        if self.optim == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.decay)

        self.set_scheduler()

        if self.resume:
            self.load_last_state()

        self.wandb_config = self._create_config()
        self._initiate_wandb_run()

    def root_mean_squared_error(self, pred_force, target_force, pred_torque,
                                target_torque):
        ## ROOT MEAN SQUARED ERROR
        return torch.sqrt(
            torch.mean((pred_force - target_force) ** 2)) + torch.sqrt(
            torch.mean((pred_torque - target_torque) ** 2))

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
        model = ParticleTorForPredictor_V1(self.in_dim,
                                           force_net_config=self.force_net_config,
                                           torque_net_config=self.torque_net_config,
                                           ellipsoid_config=self.ellipsoid_config,
                                           dropout=self.dropout,
                                           batch_norm=self.batch_norm,
                                           device=self.device,
                                           initial_weights=self.initial_weight).to(
            self.device)
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
            "overfit": self.overfit,
            "shrink": self.shrink,
            "batch_size": self.batch_size,
            "train_idx": self.train_idx,
            "lr": self.lr,
            "in_dim": self.in_dim,
            "force-net_config": self.force_net_config,
            "torque-net_config": self.torque_net_config,
            "ellipsoid_config": self.ellipsoid_config,
            "batch_norm": self.batch_norm,
            "initial_weight": self.initial_weight,
            "optim": self.optim,
            "decay": self.decay,
            "use_scheduler": self.use_scheduler,
            "scheduler_type": self.scheduler_type,
            "scheduler_patience": self.scheduler_patience,
            "scheduler_threshold": self.scheduler_threshold,
            "loss_type": self.loss_type,
            "clipping": self.clipping,
            "resume": self.resume,
        }
        print("***************** Config *******************")
        print(config)

        return config

    def _initiate_wandb_run(self):
        if self.log:
            self.wandb_config = self._create_config()

            self.wandb_run = wandb.init(project=self.project,
                                        group=self.group,
                                        config=self.wandb_config)
            self.wandb_run_name = self.wandb_run.name
            self.wandb_run_path = self.wandb_run.path
            self.wandb_run.summary["job_id"] = self.job_id
            self.wandb_run.summary["data_path"] = self.data_path
            self.wandb_run.summary["train_idx"] = self.train_idx
            self.wandb_run.summary["in_dim"] = self.in_dim
            self.wandb_run.summary["train_size"] = self.train_size
            self.wandb_run.summary["valid_size"] = self.valid_size

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
        batch_counter = 0
        for i, (
                (dr, orientation, n_orientation)
                , target_force, target_torque, energy) in enumerate(
            self.train_dataloader):
            batch_counter += 1
            self.optimizer.zero_grad()
            dr.requires_grad = True
            orientation.requires_grad = True
            dr = dr.to(self.device)
            orientation = orientation.to(self.device)
            n_orientation = n_orientation.to(self.device)

            predicted_force, predicted_torque = self.model(
                dr, orientation, n_orientation)

            target_force = target_force.to(self.device)
            target_torque = target_torque.to(self.device)

            _loss = self.loss(predicted_force, target_force, predicted_torque,
                              target_torque)

            train_loss += _loss.item()
            running_loss += _loss.item()

            _loss.backward()
            if self.clipping:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            del dr, orientation, n_orientation, target_force, target_torque, energy
            if False and i % 500 == 499:
                # if self.log:
                #     wandb.log({'running_loss': running_loss / 99.,
                #                })
                print('running_loss: ', running_loss / 499.)
                running_loss = 0.0
            torch.cuda.empty_cache()
        train_loss = train_loss / batch_counter

        return train_loss

    def _validation(self, data_loader, print_output=False):
        self.model.eval()
        # with torch.no_grad():
        total_error = 0.
        batch_counter = 0
        for i, (
                (dr, orientation, n_orientation)
                , target_force, target_torque, energy) in enumerate(
            data_loader):
            batch_counter += 1
            dr.requires_grad = True
            orientation.requires_grad = True
            dr = dr.to(self.device)
            orientation = orientation.to(self.device)
            n_orientation = n_orientation.to(self.device)

            predicted_force, predicted_torque = self.model(
                dr, orientation, n_orientation)

            target_force = target_force.to(self.device)

            target_torque = target_torque.to(self.device)

            _error = self.root_mean_squared_error(predicted_force.detach(),
                                                  target_force,
                                                  predicted_torque.detach(),
                                                  target_torque)

            total_error += _error.item()

            if print_output and i % 200 == 0:
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                print("force prediction: ", predicted_force[0])
                print("force target: ", target_force[0])
                print("torque prediction: ", predicted_torque[0])
                print("torque target: ", target_torque[0])
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            del dr, orientation, n_orientation, target_force, target_torque, energy, predicted_force, predicted_torque
            torch.cuda.empty_cache()

        return total_error / batch_counter

    def run(self):
        if self.overfit:
            self._run_overfit()
        else:
            self._run_train_valid()

    def _run_overfit(self):
        if self.log:
            wandb.watch(models=self.model, criterion=self.loss,
                        log="gradients",
                        log_freq=10)
        print(
            '**************************Overfitting*******************************')

        for epoch in range(self.epochs):
            train_loss = self._train()
            if epoch % 5 == 0:
                print('####### epoch {}/{}: \n\t train_loss: {}'.
                      format(epoch + 1, self.epochs, train_loss))
                if self.log:
                    wandb.log({'train_loss': train_loss})
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
            wandb.watch(models=self.model,
                        criterion=self.loss,
                        log="gradients",
                        log_freq=10)

        print(
            '**************************Training*******************************')
        self.best_val_error = None

        for epoch in range(self.epochs):

            train_loss = self._train()
            if epoch % 10 == 0:
                val_error = self._validation(
                    self.valid_dataloader, print_output=True)
                if self.use_scheduler:
                    self.scheduler.step(val_error)
                print('epoch {}/{}: \n\t train_loss: {}, \n\t val_error: {}'.
                      format(epoch + 1, self.epochs, train_loss, val_error))
                if self.log:
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
        # print(
        #     '**************************Testing*******************************')
        # self.test_error, self.test_force_error, self.test_torque_error = self._validation(
        #     self.test_dataloader,
        #     print_output=True)
        # print('Testing \n\t test error: {}'.
        #       format(self.test_error))
        # if self.log:
        #     self.wandb_run.summary['test error'] = self.test_error
        #     wandb.finish()
        # checkpoint = {
        #     'epoch': epoch,
        #     'model': self.model.state_dict(),
        #     'optimizer': self.optimizer.state_dict()
        # }
        # torch.save(checkpoint, 'last_checkpoint.pth')
