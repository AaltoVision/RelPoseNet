import hydra
from experiments.seven_scenes.pipeline import SevenScenesBenchmark


@hydra.main(config_path="configs", config_name="main")
def main(cfg):
    benchmark = None
    if cfg.experiment.experiment_params.name == '7scenes':
        benchmark = SevenScenesBenchmark(cfg)

    benchmark.evaluate()


if __name__ == "__main__":
    main()
