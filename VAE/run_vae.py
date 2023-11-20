import torch
from torch.optim import Adam
import torch.nn as nn
import os
import sys
from vae.VAE import VAE
from vae.encoders import Encoder_conv_shallow, Encoder_conv_deep
from vae.decoders import Decoder_circular_conv_shallow2, Decoder_circular_conv_deep
from utils_vae.dataloader import load_LiDARDataset, concat_csvs
from VAE.utils_vae.plotting_vae import *
from trainer import Trainer
from tester import Tester
import numpy as np
import argparse


# HYPERPRAMETERS
LEARNING_RATE = 0.001
N_EPOCH = 25
BATCH_SIZE = 64     
LATENT_DIMS = 12

def main(args):
    # Set hyperparameters
    BATCH_SIZE = args.batch_size        # Default: 64
    N_EPOCH = args.epochs               # Default: 25
    LATENT_DIMS = args.latent_dims      # Default: 12
    LEARNING_RATE = args.learning_rate  # Default: 0.001
    NUM_SEEDS = args.num_seeds          # Default: 1
    BETA = args.beta                    # Default: 1
    EPS_WEIGHT = args.eps_weight        # Default: 1
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Data paths
    path_empty = "data/LiDAR_synthetic_empty.csv"
    path_moving_dense = "data/LiDAR_synthetic_onlyMovingObst_dense.csv"
    path_moving_sparse = "data/LiDAR_synthetic_onlyMovingObst_sparse.csv"
    path_static_dense = "data/LiDAR_synthetic_onlyStaticObst_dense.csv"
    path_static_moving = "data/LiDAR_synthetic_staticMovingObst.csv"
    path_moving_obs_no_rules = "data/LiDAR_MovingObstaclesNoRules.csv"

    DATA_PATHS = [path_moving_dense, path_static_dense, path_static_moving, path_moving_obs_no_rules]

    concat_path = 'data/concatinated_data.csv'
    #concat_csvs(DATA_PATHS, concat_path)
    
    # Load data
    datapath = concat_path 
    rolling_degrees = [20,-20]
    num_rotations = 2000
    _, _, _, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(datapath,  
                                                                                mode='max', 
                                                                                batch_size=BATCH_SIZE, 
                                                                                train_test_split=0.7,
                                                                                train_val_split=0.3,
                                                                                shuffle=True,
                                                                                extend_dataset_roll=True,
                                                                                num_rotations=num_rotations,
                                                                                roll_degrees=rolling_degrees,
                                                                                add_noise_to_train=True)
    #datapath = 'data/LiDAR_MovingObstaclesNoRules.csv'
    if args.mode == 'train':
        # Set global model name 
        name = args.model_name
        model_name_ = f'{name}_latent_dims_{LATENT_DIMS}'

        # Create Variational Autoencoder(s)
        if args.model_name == 'ShallowConvVAE':
            encoder = Encoder_conv_shallow(latent_dims=LATENT_DIMS, eps_weight=EPS_WEIGHT)
            decoder = Decoder_circular_conv_shallow2(latent_dims=LATENT_DIMS)
            vae = VAE(encoder=encoder, decoder=decoder, latent_dims=LATENT_DIMS).to(device)

        if args.model_name == 'DeepConvVAE':
            encoder = Encoder_conv_deep(latent_dims=LATENT_DIMS, eps_weight=EPS_WEIGHT)
            decoder = Decoder_circular_conv_deep(latent_dims=LATENT_DIMS)
            vae = VAE(encoder=encoder, decoder=decoder, latent_dims=LATENT_DIMS).to(device)

        # Train model
        optimizer = Adam(vae.parameters(), lr=LEARNING_RATE) # TODO: make this an argument, and make it possible to choose between Adam and SGD
        trainer = Trainer(model=vae, 
                            epochs=N_EPOCH,
                            learning_rate=LEARNING_RATE,
                            batch_size=BATCH_SIZE,
                            dataloader_train=dataloader_train,
                            dataloader_val=dataloader_val,
                            optimizer=optimizer,
                            beta=BETA)
        
        trainer.train()

        # Save model
        if args.save_model:
            print(f'Saving model to path: models/~/{model_name_}.json...\n')
            vae.encoder.save(path=f'models/encoders/{model_name_}.json')
            vae.decoder.save(path=f'models/decoders/{model_name_}.json')

        # Plotting
        if 'reconstructions' in args.plot:
            plot_reconstructions(model=trainer.model, 
                                dataloader=dataloader_test, 
                                model_name_=model_name_, 
                                device=device, 
                                num_examples=args.num_examples, 
                                save=True, 
                                loss_func=trainer.loss_function)
        
        if 'latent_distributions' in args.plot:
            model_name_latdist = f'{model_name_}_beta_{BETA}_latent_distributions'
            plot_latent_distributions(model=vae, 
                                      dataloader=dataloader_test, 
                                      model_name=model_name_latdist,
                                      device=device,
                                      num_examples=args.num_examples, 
                                      save=True)
        
        tester = Tester(model=vae, dataloader_test=dataloader_test, trainer=trainer)
        print(f'Test loss: {tester.test()}')
        
        if any(mode in args.plot for mode in ['loss', 'separated_losses']): # Plotting modes that possibly need trainings across multiple seeds
            print("Staring multiple seeds training og plotting loss...")
            # CREATE MULTIPLE MODELS WITH DIFFERENT RANDOM SEEDS, TRAIN AND PLOT TOTAL, BCE AND KL LOSSES
            total_train_losses = np.zeros((NUM_SEEDS, N_EPOCH))
            total_val_losses = np.zeros((NUM_SEEDS, N_EPOCH))
            bce_train_losses = np.zeros((NUM_SEEDS, N_EPOCH))
            bce_val_losses = np.zeros((NUM_SEEDS, N_EPOCH))
            kl_train_losses = np.zeros((NUM_SEEDS, N_EPOCH))
            kl_val_losses = np.zeros((NUM_SEEDS, N_EPOCH))
            
            test_losses = np.zeros(NUM_SEEDS)
            for i in range(NUM_SEEDS):
                print(f'Random seed {i+1}/{NUM_SEEDS}')
                # Generate different initialization of training and validation data for the new trainer
                _, _, _, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(datapath,  
                                                                                            mode='max', 
                                                                                            batch_size=BATCH_SIZE, 
                                                                                            train_test_split=0.7,
                                                                                            train_val_split=0.3,
                                                                                            shuffle=True,
                                                                                            extend_dataset_roll=True,
                                                                                            roll_degrees=rolling_degrees,
                                                                                            add_noise_to_train=True)
                # Create Variational Autoencoder(s)
                if args.model_name == 'ShallowConvVAE':
                    encoder = Encoder_conv_shallow(latent_dims=LATENT_DIMS, eps_weight=EPS_WEIGHT)
                    decoder = Decoder_circular_conv_shallow2(latent_dims=LATENT_DIMS)
                if args.model_name == 'DeepConvVAE':
                    encoder = Encoder_conv_deep(latent_dims=LATENT_DIMS, eps_weight=EPS_WEIGHT)
                    decoder = Decoder_circular_conv_deep(latent_dims=LATENT_DIMS)

                vae = VAE(encoder=encoder, decoder=decoder, latent_dims=LATENT_DIMS).to(device)
                optimizer = Adam(vae.parameters(), lr=LEARNING_RATE)
                
                trainer = Trainer(model=vae, 
                            epochs=N_EPOCH,
                            learning_rate=LEARNING_RATE,
                            batch_size=BATCH_SIZE,
                            dataloader_train=dataloader_train,
                            dataloader_val=dataloader_val,
                            optimizer=optimizer,
                            beta=BETA)
                trainer.train()

                total_train_losses[i,:] = trainer.training_loss['Total loss']
                total_val_losses[i,:] = trainer.validation_loss['Total loss']
                bce_train_losses[i,:] = trainer.training_loss['Reconstruction loss']
                bce_val_losses[i,:] = trainer.validation_loss['Reconstruction loss']
                kl_train_losses[i,:] = trainer.training_loss['KL divergence loss']
                kl_val_losses[i,:] = trainer.validation_loss['KL divergence loss']
                
                # Get test error anyways
                tester = Tester(model=vae, dataloader_test=dataloader_test, trainer=trainer)
                test_loss, _, _ = tester.test()
                print(f'Test loss seed {i}: {test_loss}')
                test_losses[i] = test_loss
                
                del trainer, vae, encoder, decoder, optimizer 

            total_losses = [total_train_losses, total_val_losses]
            bce_losses = [bce_train_losses, bce_val_losses]
            kl_losses = [kl_train_losses, kl_val_losses]
            labels = ['Training loss', 'Validation loss']

            if 'separated_losses' in args.plot:
                plot_separated_losses(total_losses, bce_losses, kl_losses, labels, model_name=model_name_, save=True)   

            if 'loss' in args.plot:
                if args.num_seeds == 1:
                    plot_loss(training_loss=total_train_losses[0], validation_loss=total_val_losses[0], save=True, model_name=model_name_)
                else:
                    plot_loss_multiple_seeds(loss_trajectories=total_losses, labels=labels, model_name=model_name_, save=True)
            
            if 'test_loss_report' in args.plot:
                metadata = f'Number of seeds: {NUM_SEEDS}, epochs: {N_EPOCH}, batch size: {BATCH_SIZE}, learning rate: {LEARNING_RATE}\n'
                tester.report_test_stats(test_losses=test_losses, model_name=model_name_, metadata=metadata)

            
        if 'latent_dims_sweep' in args.plot: 
            print("Staring latent dimension size sweep...")
            model_name_ = f'{name}_latent_dims_sweep_2'
            latent_dims_grid = [1, 2, 6, 12, 24]
            total_val_losses_for_latent_dims = [] # Fill up with val losses for each latent dim
            bce_val_losses_for_latent_dims = [] # Fill up with val losses for each latent dim
            kl_val_losses_for_latent_dims = [] # Fill up with val losses for each latent dim
            for l in latent_dims_grid:
                print(f'Latent dimension: {l}')
                total_val_losses = np.zeros((NUM_SEEDS, N_EPOCH))
                bce_val_losses = np.zeros((NUM_SEEDS, N_EPOCH))
                kl_val_losses = np.zeros((NUM_SEEDS, N_EPOCH))

                test_losses = np.zeros(NUM_SEEDS)

                for i in range(NUM_SEEDS):
                    print(f'Random seed {i+1}/{NUM_SEEDS}')
                    # Generate different initialization of training and validation data for the new trainer
                    _, _, _, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(datapath,  
                                                                                                mode='max', 
                                                                                                batch_size=BATCH_SIZE, 
                                                                                                train_test_split=0.7,
                                                                                                train_val_split=0.3,
                                                                                                shuffle=True,
                                                                                                extend_dataset_roll=True,
                                                                                                roll_degrees=rolling_degrees,
                                                                                                add_noise_to_train=True)
                    # Create Variational Autoencoder(s)
                    if args.model_name == 'ShallowConvVAE':
                        encoder = Encoder_conv_shallow(latent_dims=l, eps_weight=EPS_WEIGHT)
                        decoder = Decoder_circular_conv_shallow2(latent_dims=l)
                    if args.model_name == 'DeepConvVAE':
                        encoder = Encoder_conv_deep(latent_dims=l, eps_weight=EPS_WEIGHT)
                        decoder = Decoder_circular_conv_deep(latent_dims=l)

                    vae = VAE(encoder=encoder, decoder=decoder, latent_dims=l).to(device)
                    optimizer = Adam(vae.parameters(), lr=LEARNING_RATE)
                    
                    trainer = Trainer(model=vae, 
                                    epochs=N_EPOCH,
                                    learning_rate=LEARNING_RATE,
                                    batch_size=BATCH_SIZE,
                                    dataloader_train=dataloader_train,
                                    dataloader_val=dataloader_val,
                                    optimizer=optimizer,
                                    beta=BETA)
                    trainer.train()

                    total_val_losses[i,:] = trainer.validation_loss['Total loss']
                    bce_val_losses[i,:] = trainer.validation_loss['Reconstruction loss']
                    kl_val_losses[i,:] = trainer.validation_loss['KL divergence loss']

                    # Get test error 
                    tester = Tester(model=vae, dataloader_test=dataloader_test, trainer=trainer)
                    test_loss, _, __build_class__ = tester.test()
                    print(f'Test loss seed {i}: {test_loss}')
                    test_losses[i] = test_loss

                    del trainer, vae, encoder, decoder, optimizer  

                many_traj_labels = [f'Seed {i+1}' for i in range(NUM_SEEDS)]
                plot_many_loss_traj(loss_trajectories=total_val_losses, labels=many_traj_labels, model_name=f'{model_name_}_latent_dim_{l}', save=True)

                total_val_losses_for_latent_dims.append(total_val_losses)
                bce_val_losses_for_latent_dims.append(bce_val_losses)
                kl_val_losses_for_latent_dims.append(kl_val_losses)

                if 'test_loss_report' in args.plot:
                    model_name_ = f'{model_name_}_latsize_{l}'
                    metadata = f'Latent dim: {l}, number of seeds: {NUM_SEEDS}, epochs: {N_EPOCH}, batch size: {BATCH_SIZE}, learning rate: {LEARNING_RATE}'
                    tester.report_test_stats(test_losses=test_losses, model_name=model_name_, metadata=metadata)

            labels = [f'Latent dim = {l}' for l in latent_dims_grid]
            plot_loss_multiple_seeds(loss_trajectories=total_val_losses_for_latent_dims, labels=labels, model_name=model_name_, save=True) # vurdere å slitte opp?
            plot_separated_losses(total_val_losses_for_latent_dims, bce_val_losses_for_latent_dims, kl_val_losses_for_latent_dims, labels, model_name=model_name_, save=True) 
          
            
        if 'eps_weight_sweep' in args.plot:
            model_name_ = f'{name}_eps_weight_sweep'
            eps_weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            
            all_test_losses = np.zeros((NUM_SEEDS, len(eps_weights)))
            all_test_losses_bce = np.zeros((NUM_SEEDS, len(eps_weights)))
            all_test_losses_kl = np.zeros((NUM_SEEDS, len(eps_weights)))
            
            print('Starting eps_weight sweep...')
            for e_idx, e in enumerate(eps_weights):
                test_losses = np.zeros(NUM_SEEDS)
                for i in range(NUM_SEEDS):
                    print(f'Random seed {i+1}/{NUM_SEEDS}')
                    # Generate different initialization of training and validation data for the new trainer
                    _, _, _, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(datapath,  
                                                                                                mode='max', 
                                                                                                batch_size=BATCH_SIZE, 
                                                                                                train_test_split=0.7,
                                                                                                train_val_split=0.3,
                                                                                                shuffle=True,
                                                                                                extend_dataset_roll=True,
                                                                                                roll_degrees=rolling_degrees,
                                                                                                add_noise_to_train=True)
                    # Create Variational Autoencoder(s)
                    if args.model_name == 'ShallowConvVAE':
                        encoder = Encoder_conv_shallow(latent_dims=LATENT_DIMS, eps_weight=e)
                        decoder = Decoder_circular_conv_shallow2(latent_dims=LATENT_DIMS)
                    if args.model_name == 'DeepConvVAE':
                        encoder = Encoder_conv_deep(latent_dims=LATENT_DIMS, eps_weight=e)
                        decoder = Decoder_circular_conv_deep(latent_dims=LATENT_DIMS)

                    vae = VAE(encoder=encoder, decoder=decoder, latent_dims=LATENT_DIMS).to(device)
                    optimizer = Adam(vae.parameters(), lr=LEARNING_RATE)
                    
                    trainer = Trainer(model=vae, 
                                    epochs=N_EPOCH,
                                    learning_rate=LEARNING_RATE,
                                    batch_size=BATCH_SIZE,
                                    dataloader_train=dataloader_train,
                                    dataloader_val=dataloader_val,
                                    optimizer=optimizer,
                                    beta=BETA)
                    trainer.train()
                    
                    # Get test error 
                    tester = Tester(model=vae, dataloader_test=dataloader_test, trainer=trainer)
                    test_loss, test_loss_bce, test_loss_kl = tester.test()
                    print(f'Test loss seed {i}: {test_loss}')
                    all_test_losses[i,e_idx] = test_loss
                    all_test_losses_bce[i,e_idx] = test_loss_bce
                    all_test_losses_kl[i,e_idx] = test_loss_kl
                    
                    # Plot some reconstructions for each model 
                    model_name_inner = f'{model_name_}_eps_{e}_seed_{i}'
                    """
                    plot_reconstructions(model=trainer.model, 
                                         dataloader=dataloader_test, 
                                         model_name_=model_name_inner, 
                                         num_examples=3,
                                         device=device,
                                         save=True,
                                         loss_func=trainer.loss_function)
                    
                    plot_latent_distributions(model=trainer.model,
                                              dataloader=dataloader_test,
                                              model_name=model_name_inner,
                                              num_examples=3,
                                              save=True,
                                              device=device)
                    """
                    
                    del trainer, vae, encoder, decoder, optimizer 
                
                #all_test_losses.append(test_losses)
                #all_test_losses[,:] = test_losses
                
                metadata = f'Latent dim: {LATENT_DIMS}, number of seeds: {NUM_SEEDS}, epochs: {N_EPOCH}, batch size: {BATCH_SIZE}, learning rate: {LEARNING_RATE}\n'
                tester.report_test_stats(test_losses=test_losses, model_name=model_name_, metadata=metadata)
            
            # Plot test errors as function of the epw_weights
            plt.style.use('ggplot')
            plt.rc('font', family='serif')
            plt.rc('xtick', labelsize=12)
            plt.rc('ytick', labelsize=12)
            plt.rc('axes', labelsize=12)
            fig, ax = plt.subplots()
            x = np.arange(len(all_test_losses[0])) # num_samples

            losses = [all_test_losses, all_test_losses_bce, all_test_losses_kl]
            labels = ['Total loss', 'Reconstruction loss', 'KL divergence']
            
            for i, loss_traj in enumerate(losses):
                # Get mean and variance
                conf_interval = 1.96 * np.std(loss_traj, axis=0) / np.sqrt(len(x)) # 95% confidence interval
                mean_error_traj = np.mean(loss_traj, axis=0)
                # Insert into plot
                ax.plot(eps_weights, mean_error_traj, label=labels[i], linewidth=1)
                ax.fill_between(eps_weights, mean_error_traj - conf_interval, mean_error_traj + conf_interval, alpha=0.2)

            ax.set_xlabel('Epsilon weight')
            ax.set_ylabel('Test loss')
            ax.legend()
            
            fig.savefig(f'plots/EPS_WEIGHT_SWEEP_{model_name_}.pdf', bbox_inches='tight')
        
        
        if "latent_dist_kde" in args.plot:
            betas = [0, 0.2, 0.5, 1, 2, 5, 10]
            latent_dims_kde = 2
            _, _, _, dataloader_train, dataloader_val, dataloader_test  = load_LiDARDataset(datapath,  
                                                                                                mode='max', 
                                                                                                batch_size=BATCH_SIZE, 
                                                                                                train_test_split=0.7,
                                                                                                train_val_split=0.3,
                                                                                                shuffle=True,
                                                                                                extend_dataset_roll=True,
                                                                                                roll_degrees=rolling_degrees,
                                                                                                add_noise_to_train=True)
            
            for b in betas:
                print(f"Training with beta = {b}")
                # Create Variational Autoencoder(s)
                encoder1 = Encoder_conv_shallow(latent_dims=latent_dims_kde, eps_weight=EPS_WEIGHT)
                decoder1 = Decoder_circular_conv_shallow2(latent_dims=latent_dims_kde)
                encoder2 = Encoder_conv_deep(latent_dims=latent_dims_kde, eps_weight=EPS_WEIGHT)
                decoder2 = Decoder_circular_conv_deep(latent_dims=latent_dims_kde)

                vae1 = VAE(encoder=encoder1, decoder=decoder1, latent_dims=latent_dims_kde).to(device)
                optimizer1 = Adam(vae1.parameters(), lr=LEARNING_RATE)
                vae2 = VAE(encoder=encoder2, decoder=decoder2, latent_dims=latent_dims_kde).to(device)
                optimizer2 = Adam(vae2.parameters(), lr=LEARNING_RATE)
                
                trainer1 = Trainer(model=vae1, 
                                epochs=N_EPOCH,
                                learning_rate=LEARNING_RATE,
                                batch_size=BATCH_SIZE,
                                dataloader_train=dataloader_train,
                                dataloader_val=dataloader_val,
                                optimizer=optimizer1,
                                beta=b)
                trainer1.train()
                trainer2 = Trainer(model=vae2, 
                                epochs=N_EPOCH,
                                learning_rate=LEARNING_RATE,
                                batch_size=BATCH_SIZE,
                                dataloader_train=dataloader_train,
                                dataloader_val=dataloader_val,
                                optimizer=optimizer2,
                                beta=b)
                trainer2.train()
                
                # Run on whole test set to obtain kde of latent space
                latent_space_kde(model=vae1, dataloader=dataloader_test, name=f'latent_kde_beta_{b}_shallow', save=True)
                latent_space_kde(model=vae2, dataloader=dataloader_test, name=f'latent_kde_beta_{b}_deep', save=True)

                
                del trainer1, vae1, encoder1, decoder1, optimizer1, trainer2, vae2, encoder2, decoder2, optimizer2
                
                

                            
             
        




"""
    if args.mode == 'test':
        '''vae_path = f'models/vaes/{args.model_name_load}.json'
        encoder_path = f'models/encoders/{args.model_name_load}.json'
        vae = VAE().load(path=vae_path) 
        encoder = vae.encoder'''

        # Generate test error for all the models in vae folder for each latent dim, averaged over NUM_SEEDS
        latent_dims_grid = [1, 2, 4, 8, 12]
        seeds = list(range(1, NUM_SEEDS+1))
        test_losses = np.zeros((len(latent_dims_grid), NUM_SEEDS))
        for l in latent_dims_grid:
            for seed in seeds:
                name = "ShallowConvVAE"
                model_name_ = f'{name}_latent_dims_sweep'
                path = f'models/vaes/{model_name_}_latent_dim_{l}_seed_{seed}.json'
                encoder = Encoder_conv_shallow(latent_dims=l)
                decoder = Decoder_circular_conv_shallow2(latent_dims=l)
                vae = VAE_integrated_shallow(latent_dims=l).load(path=path)
                # Obtain test loss for this latent dim and seed combination
                for x_batch in dataloader_test:
                    x_batch = x_batch.to(device)
                    x_hat, mu, log_var = vae(x_batch)
                    loss, _, _ = trainer.loss_function(x_hat, x_batch, mu, log_var, beta=1)
                    test_losses[l, seed-1] += loss.item()
                test_losses[l, seed-1] /= len(dataloader_test.dataset)
        plot_test_error_as_function_of(test_errors=test_losses, variable=latent_dims_grid, variable_name='Latent dimension', save=True, model_name=model_name_)
"""




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',
                        help= 'Progam mode',
                        choices=['train', 'test']
    )
    parser.add_argument('--model_name',
                        help= 'Name of model to train',
                        type=str,
                        choices=['ShallowConvVAE', 'DeepConvVAE'],
                        default='ShallowConvVAE'
    )
    parser.add_argument('--model_name_load',
                        help= 'Name of model to be used for testing',
                        type=str,
                        default='ShallowConvVAE'
    )
    parser.add_argument('--plot',
                        help= 'Plotting mode',
                        type=str,
                        choices=['reconstructions', 'loss', 'separated_losses', 'latent_dims_sweep', 'latent_distributions', 'test_loss_report', 'eps_weight_sweep', 'latent_dist_kde'],
                        nargs='+'
    )
    parser.add_argument('--save_model',
                        help= 'Save model',
                        type=bool,
                        default=False
    )
    parser.add_argument('--num_seeds',
                        help= 'Number of seeds to train and plot',
                        type=int,
                        default=1
    )
    parser.add_argument('--beta',
                    help= 'beta for beta-VAE. Default 1: vanilla VAE',
                    type=float,
                    default=1
    )
    parser.add_argument('--num_examples',
                        help= 'Number of reconstruction examples to plot',
                        type=int,
                        default=10
    )
    parser.add_argument('--batch_size', 
                        help= 'Batch size for training', 
                        type=int, 
                        default=64
    )
    parser.add_argument('--epochs',
                        help= 'Number of epochs for training', 
                        type=int, 
                        default=25
    )
    parser.add_argument('--datapath', 
                        type=str, 
                        default='data/LiDAR_MovingObstaclesNoRules.csv'
    )
    parser.add_argument('--learning_rate',
                        help= 'Learning rate for training', 
                        type=float, 
                        default=0.001
    )
    parser.add_argument('--latent_dims',
                        help= 'Number of latent dimensions', 
                        type=int, 
                        default=12
    )
    parser.add_argument('--eps_weight',
                        help= 'Weight of noise in loss function', 
                        type=float, 
                        default=1
    )


    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print('\n\nKeyboard interrupt detected, exiting.')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    finally:
        print('Done.')