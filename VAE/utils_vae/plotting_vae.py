import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
import pandas as pd
import seaborn as sns
import scipy.stats as stats


def plot_loss(training_loss, validation_loss, model_name, save=False) -> None:
    """Plots training and validation loss as functions of number of epochs (only for one seed/loss-trajectory)"""
    print('Plotting loss for one seed only...')
    epochs = np.arange(len(training_loss))
    plt.style.use('ggplot')
    #plt.style.use('seaborn-poster')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    fig, ax = plt.subplots()
    ax.plot(epochs, training_loss, label='Training_loss')
    ax.plot(epochs, validation_loss, label='Validation loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    if save:
        plt.savefig(f'plots/LOSS_SINGLESEED_{model_name}.pdf', bbox_inches='tight')


def plot_reconstruction(index:int, X:torch.Tensor, X_hat:torch.Tensor, model_name:str, save=False, loss:tuple=None) -> None:
    """Plots a single top-down view of the range data X and the reconstructed range data X_hat, as well as the loss for that sample"""
    angles = np.linspace(0, 2*np.pi, 180) - np.pi / 2
    X = np.ones_like(X) - X # we need x = x/150 so we invert the normalization from load_LiDARDataset() since 1-x = x/150
    X_hat = np.ones_like(X_hat) - X_hat

    plt.style.use('ggplot')
    #plt.style.use('seaborn-poster')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.scatter(angles, X_hat, s=1.2)
    ax.scatter(angles, X, s=1.2)
    ax.legend(['Reconstructed', 'True'], loc='upper right', bbox_to_anchor=(1.2, 1.1))
    if loss:
        tot_loss, bce_loss, kl_divergence = loss
        ax.text(-0.35, -0.05, f'Total loss = {tot_loss:.3f}\nReconstruction loss = {bce_loss:.3f}\nKL-divergence = {kl_divergence:.3f}', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    if save:
        plt.savefig(f'plots/RECONSTRUCTION_{model_name}_{index}.pdf')


def plot_reconstructions(model:nn.Module, dataloader:torch.utils.data.DataLoader, model_name_:str, device:str, num_examples:int=7, save=True, loss_func=None) -> None:
    """Plots num_examples reconstructions from the given dataloader, data is assumed shuffled"""
    print('Plotting reconstructions...')
    batches_to_sample_from = [i for i in range(num_examples)] # dataloader.dataset is shuffeled so these are random samples
    for batch_idx, x_batch in enumerate(dataloader):
        if batch_idx not in batches_to_sample_from:
            continue
        else:
            x = x_batch.to(device)
            x_hat, mu, sigma = model(x)
            beta = model.beta
            tot_loss, bce_loss, kl_divergence = loss_func(x_hat, x, mu, sigma, beta, N=len(dataloader.dataset), M=len(dataloader))
            
            x = x.cpu().detach().numpy()[0,0,:]
            x_hat = x_hat.cpu().detach().numpy()[0,0,:]
            tot_loss = tot_loss.cpu().detach().numpy()
            bce_loss = bce_loss.cpu().detach().numpy()
            kl_divergence = kl_divergence.cpu().detach().numpy()

            loss = (tot_loss, bce_loss, kl_divergence)

            plot_reconstruction(index=batch_idx, X=x, X_hat=x_hat, model_name=model_name_, save=True, loss=loss)


def plot_latent_distributions(model:nn.Module, dataloader:torch.utils.data.DataLoader, model_name:str, device:str, num_examples:int=7, save=True) -> None:
    """Plots num_examples latent distributions from the given dataloader, data is assumed shuffled"""
    print('Plotting latent distributions...')
    latent_representations = np.array([])
    labels = np.array(range(1,model.latent_dims+1))
    for i, x_batch in enumerate(dataloader):
        # Get latent representation
        x_hat, mu, sigma = model(x_batch)
        mu, log_var, z = model.encoder(x_batch)
        sigma = torch.exp(0.5*log_var) 
        
        x = x_batch.cpu().detach().numpy()[0,0,:]
        x_hat = x_hat.cpu().detach().numpy()[0,0,:]
        z = z.detach().numpy().flatten()
        mu = mu.detach().numpy().flatten()
        sigma = sigma.detach().numpy().flatten()

        # Plot latent representation
        plt.style.use('ggplot')
        plt.rc('font', family='serif')
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.rc('axes', labelsize=12)

        # Input-reconstruction scatterplot
        angles = np.linspace(0, 2*np.pi, 180) - np.pi / 2
        X = np.ones_like(x) - x # invert for plotting 
        X_hat = np.ones_like(x_hat) - x_hat 
        fig = plt.figure(figsize=(17,5))
        ax1 = fig.add_subplot(121,projection='polar')
        
        ax1.scatter(angles, X_hat, s=1.2)
        ax1.scatter(angles, X, s=1.2)
        ax1.legend(['Reconstructed', 'True'], loc='upper right', bbox_to_anchor=(1.2, 1.1))
        

        # Latent-distribution subplot
        ax2 = fig.add_subplot(122)
        x = np.linspace(min(mu) - 3*sigma[np.argmin(mu)], max(mu) + 3*sigma[np.argmax(mu)], 100)
        for k in range(len(mu)):
            if k > 6: # 6 is number of colors in ggplot
                lin = '--'
            else:
                lin = '-'
            ax2.plot(x, stats.norm.pdf(x, mu[k], sigma[k]), label=f'z{k+1}', linestyle=lin, linewidth=1.5)
        ax2.legend()
        ax2.set_ylabel('Probability density')

        plt.savefig(f'plots/{model_name}_number_{i}.pdf', bbox_inches='tight')
        
        if i > num_examples:
            break

def latent_space_kde(model:nn.Module, dataloader:torch.utils.data.DataLoader, name:str, save=True):
    """Run test set through encoder of vae and plot kde in 2D of latent distributions. Assumes latent dim of model is 2"""
    zs = np.zeros((1,2))
    for i, x_batch in enumerate(dataloader):
        # Get latent representation
        _, _, z = model.encoder(x_batch) # z is assumed 2D
        z = z.detach().numpy()[0,:,:]
        if i == 0:
            zs = z
        else:
            zs = np.vstack((zs, z))
    
    # create kde plot
    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    sns.kdeplot(x=zs[:,0], y=zs[:,1], fill=True, levels=20)
    plt.xlabel('z1')
    plt.ylabel('z2')
    ax = plt.gca()
    ax.set_box_aspect(1)
    #ax.set_aspect('equal', adjustable='box')
    #ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.savefig(f'plots/{name}.pdf', bbox_inches='tight')
    plt.clf()
    plt.cla()
        
        

def plot_loss_multiple_seeds(loss_trajectories:list, labels:list, model_name:str, save=False) -> None:
    """
    Plots arbitrary loss trajectories averaged over multiple seeds as functions of number of epochs, including variance-bands
    loss_trajectories must be a list with entries as np.ndarray:(n_seeds, n_epochs)
    """
    print('Plotting loss trajectories across multiple seeds...')
    fill_between = True
    if isinstance(loss_trajectories, np.ndarray): # force list-type if only one trajectory, idk
        loss_trajectories = [loss_trajectories]

    plt.style.use('ggplot')
    #plt.style.use('seaborn-poster')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    fig, ax = plt.subplots()
    x = np.arange(len(loss_trajectories[0][0,:])) # epochs, extracted from first trajectory
    for i, loss_traj in enumerate(loss_trajectories):
        # Get mean and variance
        conf_interval = 1.96 * np.std(loss_traj, axis=0) / np.sqrt(len(x)) # 95% confidence interval
        mean_error_traj = np.mean(loss_traj, axis=0)
        variance_traj = np.var(loss_traj, axis=0)
        # Insert into plot
        ax.plot(x, mean_error_traj, label=labels[i], linewidth=1)
        ax.fill_between(x, mean_error_traj - conf_interval, mean_error_traj + conf_interval, alpha=0.2)

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    if save:
        fig.savefig(f'plots/LOSS_MULTISEED_{model_name}.pdf', bbox_inches='tight')
    
def plot_many_loss_traj(loss_trajectories:np.ndarray, model_name:str, labels, save=False) -> None:
    """
    Plots arbitrary loss trajectories on top of each other
    loss_trajectories must be ndarray:(n_trajectories, n_epochs)
    """
    plt.style.use('ggplot')
    #plt.style.use('seaborn-poster')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    fig, ax = plt.subplots()
    x = np.arange(len(loss_trajectories[0,:])) # epochs, extracted from first trajectory
    for i, loss_traj in enumerate(loss_trajectories):
        ax.plot(x, loss_traj, label=labels[i], linewidth=1)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    if save:
        fig.savefig(f'plots/LOSS_MULTIPLETRAJS_{model_name}.pdf', bbox_inches='tight')



def plot_separated_losses(total_losses:list, BCE_losses:list, KL_losses:list, labels:list, model_name:str, save=False) -> None:
    """
    Plots seperated losses as [total, BCE, KL] averaged over multiple seeds as functions of number of epochs, including variance-bands
    all three loss_trajectories must be lists with entries as np.ndarray:(n_seeds, n_epochs)
    """
    print('Plotting separated loss trajectories across multiple seeds...')
    
    fill_between = True
    if isinstance(KL_losses, np.ndarray): # force list-type if only one trajectory
        KL_losses = [KL_losses]
    if isinstance(BCE_losses, np.ndarray):
        BCE_losses = [BCE_losses]
    if isinstance(total_losses, np.ndarray):
        total_losses = [total_losses]
    
    mapping = {0:total_losses, 1:BCE_losses, 2:KL_losses}
    name = {0:'Total loss', 1:'Reconstruction error', 2:'KL divergence'}

    plt.style.use('ggplot')
    #plt.style.use('seaborn-poster')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    fig, axes = plt.subplots(1, 3, figsize=(20,5))
    x = np.arange(len(KL_losses[0][0,:])) # epochs, extracted from first trajectory
    for i in range(3):
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel('Loss')
        axes[i].set_title(name[i])
        for j, loss_traj in enumerate(mapping[i]): # go through loss trajs (np.ndarray:(n_seeds, n_epochs))
            # Get mean and variance
            mean_error_traj = np.mean(loss_traj, axis=0)
            variance_traj = np.std(loss_traj, axis=0)
            conf_interval = 1.96 * np.std(loss_traj, axis=0) / np.sqrt(len(x)) # 95% confidence interval
            # Insert into plot
            axes[i].plot(x, mean_error_traj, label=labels[j], linewidth=1)
            axes[i].fill_between(x, mean_error_traj - conf_interval, mean_error_traj + conf_interval, alpha=0.2)
            axes[i].legend()

    if save:
        fig.savefig(f'plots/LOSS_SEPARATED_{model_name}.pdf', bbox_inches='tight')

def plot_test_error_as_function_of(test_errors:np.ndarray, variable, variable_name, save, model_name): # Not in use
    """
    Plots test error as function of variable
    test_errors must be a np.ndarray: variables x seeds
    """
    print(f'Plotting test error as function of {variable_name}...')

    means = np.mean(test_errors, axis=0)
    conf_interval = 1.96 * np.std(test_errors, axis=0) / np.sqrt(len(test_errors)) # 95% confidence interval
    variances = np.std(test_errors, axis=0)

    plt.style.use('ggplot')
    #plt.style.use('seaborn-poster')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    fig, ax = plt.subplots()

    ax.plot(variable, means, label='Test error', linewidth=1)
    ax.fill_between(variable, means - variances, means + variances, alpha=0.2)
    ax.set_xlabel(f'{variable_name}')
    ax.set_ylabel('Loss')
    ax.legend()
    if save:
        plt.savefig(f'plots/TEST_ERROR_AS_FUNCTION_OF_{variable}_{model_name}.pdf', bbox_inches='tight')

