import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import hydra

from omegaconf import OmegaConf, DictConfig
from utils import get_local_dir

@hydra.main(version_base=None, config_path="config", config_name="config_uncertainty")
def main(config: DictConfig):
    # datasets = ['hh', 'shp', 'jeopardy']
    # datasets_compare = [dataset for dataset in datasets if dataset not in config.datasets]
    # with open(f'/home/scratch/vdas/dpo/uncertainties/uncertainties_{config.datasets[0]}_{config.datasets[0]}_train.pkl','rb') as f:
    #     train_uncertainties = pickle.load(f)
    # with open(f'/home/scratch/vdas/dpo/uncertainties/uncertainties_{config.datasets[0]}_{config.datasets[0]}_eval.pkl','rb') as f:
    #     eval_uncertainties = pickle.load(f)
    #
    # compare_uncertainties = []
    # for ds_name in datasets_compare:
    #     with open(f'/home/scratch/vdas/dpo/uncertainties/uncertainties_{config.datasets[0]}_{ds_name}.pkl', 'rb') as f:
    #         compare_uncertainties.append(pickle.load(f))
    #
    # sns.kdeplot(train_uncertainties, label=f'train {config.datasets[0]}')
    # sns.kdeplot(eval_uncertainties, label=f'eval {config.datasets[0]}')
    # for i, compare_uncertainty in enumerate(compare_uncertainties):
    #     sns.kdeplot(compare_uncertainty, label=f'{datasets_compare[i]}')
    # # Set xlim
    # plt.xlim(-0.005, 0.1)
    # plt.legend()
    # plt.savefig(f'uncertainty_{config.datasets[0]}.png')

    with open(f'/home/scratch/vdas/dpo/uncertainties/uncertainties_shp_shp_train_0.05_epinet.pkl','rb') as f:
        train_uncertainties = pickle.load(f)
    with open(f'/home/scratch/vdas/dpo/uncertainties/uncertainties_shp_shp_eval_0.05_epinet.pkl','rb') as f:
        eval_uncertainties = pickle.load(f)
    with open(f'/home/scratch/vdas/dpo/uncertainties/uncertainties_shp_jeopardy_0.05_epinet.pkl','rb') as f:
        jeo_uncertainties = pickle.load(f)
    sns.kdeplot(train_uncertainties, label=f'train')
    sns.kdeplot(eval_uncertainties, label=f'eval')
    sns.kdeplot(jeo_uncertainties, label=f'jeo')
    plt.legend()
    plt.savefig(f'uncertainty_shp_epinet.png')


if __name__ == '__main__':
    main()