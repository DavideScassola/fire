import pandas as pd
import os
import fnmatch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys

plt.rcParams["figure.figsize"] = (6,2.4)
plt.rcParams["figure.dpi"] = 150
plt.tight_layout()


"""
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
"""


def listdir(dir: str = './'):
    return [os.path.join(dir, f) for f in os.listdir(dir)]


def group_experiments(folders: list[str]):
    names = set(f.split('--')[0] for f in folders)
    return {n: fnmatch.filter(folders, f'{n}--*') for n in names}


def get_rewards(path: str):
    config_file = fnmatch.filter(listdir(path), '*.json')[0]
    with open(config_file, 'r') as f:
        config = json.load(f)
    rewards = config['rewards']
    if not isinstance(rewards, list):
        rewards = [rewards]
    return rewards


def get_best_model_data(name, experiments):

    df = []

    for e in experiments:
        rewards = get_rewards(e)
        dfs = pd.concat({r: pd.read_csv(os.path.join(e,
                                                     r, 'plots/best_model_info.csv'), index_col=0).iloc[-1].rename(e).to_frame().transpose() for r in rewards})

        df.append(dfs)

    df = pd.concat(df)
    df = df.reset_index(level=0)
    df = df.rename({'level_0': 'reward'}, axis='columns')

    return df


def distribution_plot(df: pd.DataFrame, name: str):
    colors = ('blue', 'green', 'red')
    fig, axs = plt.subplots()
    df_performance_test = df[['performance_test', 'reward']].pivot(
        columns=['reward'], values='performance_test')
    if len(df_performance_test) > 3:
        #boxplot = df_performance_test.boxplot()
        boxplot = plt.boxplot(df_performance_test)
        plt.set_title(
            f'Test set performance (Sharpe Ratio)\n{len(df_performance_test)} experiments')
              
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)  

        fig.savefig(f'{name}.png')


def distribution_plot(df: pd.DataFrame, name: str):
    #sns.set_theme()

    sets = ('test', 'eval', 'train')
    colors = ('blue', 'green', 'red')
    df_plot = df

    df_plot = pd.concat({c: df[[f'performance_{c}', 'reward']].rename(
        {f'performance_{c}': 'Sharpe Ratio'}, axis='columns') for c in sets})

    df_plot = df_plot.reset_index(level=0)
    df_plot.rename({'level_0': 'Dataset'}, axis='columns', inplace=True)

    samples = int(len(df)//len(set(df_plot['reward'])))

    my_pal = {'train': 'red', 'eval': 'green', 'test': 'blue'}
    if samples > 4:
        g = sns.boxplot(data=df_plot, x="reward", y="Sharpe Ratio", hue="Dataset", palette=my_pal).set_title(
            f'{name.split("/")[-1]} ({samples} experiments)')

        ax = g.axes
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, title=None)
        
        plt.ylabel("Sharpe Ratio")
        plt.grid(color='gray', linestyle='--', linewidth=1, alpha = 0.3, zorder = 0., axis='y')
        plt.title("ETH") # Change this
        #plt.legend(bbox_to_anchor=(-1., 1.1), loc='upper left', borderaxespad=0)


        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{name.split("/")[-1]}.svg')
        plt.savefig(f'plots/{name.split("/")[-1]}.png')
        df_plot.to_csv(f'plots/{name.split("/")[-1]}.csv')

    plt.cla()


def main():
    
    experiment_folder = fnmatch.filter(listdir(), '*results*experiments*')[0] if len(sys.argv)<2 else sys.argv[1]

    experiments: list = listdir(experiment_folder)

    grouped_experiments: dict = group_experiments(experiments)

    for name, experiments in grouped_experiments.items():
        assert len(experiments) > 0, f'{name}'
        df_best_model: pd.DataFrame = get_best_model_data(name, experiments)
        distribution_plot(df_best_model, name)


if __name__ == '__main__':
    main()
