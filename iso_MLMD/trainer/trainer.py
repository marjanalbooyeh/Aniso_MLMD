import torch
import torch.nn as nn
import wandb

from iso_MLMD.trainer import IsoPairNN, IsoNeighborNN, load_datasets, load_overfit_data


class IsoTrainer:
    def __init__(self, config, job_id, resume=False):
        print("***************CUDA: ", torch.cuda.is_available())
        self.resume = resume
        self.job_id = job_id
        self.project = config.project
        self.group = config.group
        self.notes = config.notes
        self.tags = config.tags
        self.wandb_log = config.wandb_log

        # dataset parameters
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.shrink = config.shrink
        self.overfit = config.overfit

        # model parameters
        self.model_type = config.model_type
        self.in_dim = config.in_dim
        self.hidden_dim = config.hidden_dim
        self.n_layer = config.n_layer
        self.box_len = config.box_len
        self.act_fn = config.act_fn
        self.dropout = config.dropout
        self.batch_norm = config.batch_norm

        # optimizer parameters
        self.optim = config.optim
        self.lr = config.lr
        self.min_lr = config.min_lr
        self.decay = config.decay
        self.use_scheduler = config.use_scheduler
        self.scheduler_type = config.scheduler_type
        self.loss_type = config.loss_type

        # prior energy parameters
        self.prior_energy = config.prior_energy
        self.prior_energy_sigma = None
        self.prior_energy_n = None
        if config.prior_energy:
            self.prior_energy_sigma = config.prior_energy_sigma
            self.prior_energy_n = config.prior_energy_n

        # run parameters
        self.epochs = config.epochs

        # select device (GPU or CPU)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # create data loaders
        if self.overfit:
            self.train_dataloader = load_overfit_data(config.data_path,
                                                      config.batch_size)
        else:
            self.train_dataloader, self.valid_dataloader, self.test_dataloader = load_datasets(
                config.data_path, config.batch_size, shrink=self.shrink)

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
            self.force_loss = nn.L1Loss().to(self.device)
            self.criteria = nn.L1Loss().to(self.device)
        elif self.loss_type == "mse":
            self.force_loss = nn.MSELoss().to(self.device)
            self.criteria = nn.MSELoss().to(self.device)
        elif self.loss_type == "huber":
            self.force_loss = nn.HuberLoss(delta=5).to(self.device)
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
        if self.wandb_log:
            self._initiate_wandb_run()

    def set_scheduler(self):
        if self.scheduler_type == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.98, patience=200, min_lr=self.min_lr,
                verbose=True)
        elif self.scheduler_type == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                             step_size=5000,
                                                             gamma=0.98,
                                                             verbose=True)

    def _create_model(self):
        if self.model_type == "pair":
            model = IsoPairNN(in_dim=self.in_dim,
                              hidden_dim=self.hidden_dim,
                              n_layers=self.n_layer,
                              act_fn=self.act_fn,
                              dropout=self.dropout,
                              batch_norm=self.batch_norm,
                              device=self.device,
                              prior_energy=self.prior_energy,
                              prior_energy_sigma=self.prior_energy_sigma,
                              prior_energy_n=self.prior_energy_n
                              )
        else:
            model = IsoNeighborNN(in_dim=self.in_dim,
                                  hidden_dim=self.hidden_dim,
                                  n_layers=self.n_layer,
                                  box_len=self.box_len,
                                  act_fn=self.act_fn,
                                  dropout=self.dropout,
                                  batch_norm=self.batch_norm,
                                  device=self.device,
                                  prior_energy=self.prior_energy,
                                  prior_energy_sigma=self.prior_energy_sigma,
                                  prior_energy_n=self.prior_energy_n
                                  )
        model.to(self.device)
        print(model)
        return model

    def load_last_state(self):
        print('reloading latest model and optimizer state...')
        last_checkpoint = torch.load('last_checkpoint.pth')
        self.model.load_state_dict(last_checkpoint['model'])
        self.optimizer.load_state_dict(last_checkpoint['optimizer'])
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.01
        print('successfully loaded model and optimizer...')

    def _create_config(self):
        config = {
            "overfit": self.overfit,
            "batch_size": self.batch_size,
            "shrink": self.shrink,
            "lr": self.lr,
            "model_type": self.model_type,
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "n_layer": self.n_layer,
            "box_len": self.box_len,
            "optim": self.optim,
            "decay": self.decay,
            "act_fn": self.act_fn,
            "dropout": self.dropout,
            "use_scheduler": self.use_scheduler,
            "scheduler_type": self.scheduler_type,
            "loss_type": self.loss_type,
            "batch_norm": self.batch_norm,
            "resume": self.resume,
            "prior_energy": self.prior_energy,
            "prior_energy_sigma": self.prior_energy_sigma,
            "prior_energy_n": self.prior_energy_n

        }
        print("***************** Config *******************")
        print(config)

        return config

    def _initiate_wandb_run(self):
        self.wandb_config = self._create_config()

        self.wandb_run = wandb.init(project=self.project, notes=self.notes,
                                    group=self.group,
                                    tags=self.tags, config=self.wandb_config)
        self.wandb_run_name = self.wandb_run.name
        self.wandb_run_path = self.wandb_run.path
        self.wandb_run.summary["job_id"] = self.job_id
        self.wandb_run.summary["data_path"] = self.data_path
        self.wandb_run.summary["in_dim"] = self.in_dim
        self.wandb_run.summary["train_size"] = self.train_size
        self.wandb_run.summary["valid_size"] = self.valid_size
        self.wandb_run.summary["test_size"] = self.test_size


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
        for i, ((x, neighbor_list), force, energy) in enumerate(
            self.train_dataloader):

            self.optimizer.zero_grad()
            x.requires_grad = True
            x = x.to(self.device)
            if self.model_type == "pair":
                predicted_force = self.model(x)
            else:
                neighbor_list = neighbor_list.to(self.device)
                predicted_force = self.model(x, neighbor_list)
            target_force = force.to(self.device)

            force_loss = self.force_loss(predicted_force, target_force)
            train_loss += force_loss.item()
            running_loss += force_loss.item()
            force_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            if i % 10 == 9:
                if self.wandb_log:
                    wandb.log({'running_loss': running_loss / 9.,
                               "learning rate": self.optimizer.param_groups[0][
                                   'lr']})
                print('running_loss: {}'.format(running_loss / 9.))

                running_loss = 0.0
                checkpoint = {'model': self.model.state_dict(),
                              'optimizer': self.optimizer.state_dict()}
                torch.save(checkpoint, 'running_checkpoint.pth')

        train_loss = train_loss / (i + 1)

        return train_loss

    def _validation(self, data_loader, print_output=False):
        self.model.eval()
        # with torch.no_grad():
        total_error = 0.
        for i, ((x, neighbor_list), force, energy) in enumerate(
            data_loader):
            x.requires_grad = True
            x = x.to(self.device)
            if self.model_type == "pair":
                predicted_force = self.model(x)
            else:
                neighbor_list = neighbor_list.to(self.device)
                predicted_force = self.model(x, neighbor_list)

            target_force = force.to(self.device)

            total_error += self.criteria(predicted_force, target_force).item()

            if print_output and i % 100 == 0:
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                print("force prediction: ", predicted_force[5][:10])
                print("force target: ", target_force[5][:10])

        return total_error / (i + 1)

    def run(self):
        if self.overfit:
            self._run_overfit()
        else:
            self._run_train_valid()

    def _run_overfit(self):
        if self.wandb_log:
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

        wandb.finish()
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, 'last_checkpoint.pth')

    def _run_train_valid(self):
        if self.wandb_log:
            wandb.watch(models=self.model, criterion=self.force_loss,
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
                val_error = self._validation(
                    self.valid_dataloader, print_output=True)
                if self.use_scheduler:
                    self.scheduler.step(val_error)
                print('epoch {}/{}: \n\t train_loss: {}, \n\t val_error: {}'.
                      format(epoch + 1, self.epochs, train_loss, val_error))
                if self.wandb_log:
                    wandb.log({'train_loss': train_loss,
                               'valid error': val_error,
                               "learning rate": self.optimizer.param_groups[0][
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
                    if self.wandb_log:
                        self.wandb_run.summary["best_epoch"] = epoch + 1
                        self.wandb_run.summary[
                            "best_val_error"] = self.best_val_error

        # Testing
        print(
            '**************************Testing*******************************')
        self.test_error = self._validation(
            self.test_dataloader,
            print_output=True)
        print('Testing \n\t test error: {}'.
              format(self.test_error))
        if self.wandb_log:
            self.wandb_run.summary['test error'] = self.test_error
            wandb.finish()
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, 'last_checkpoint.pth')
