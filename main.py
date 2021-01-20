import hydra
from relposenet.pipeline import Pipeline


@hydra.main(config_path="configs", config_name="main")
def main(cfg):
    pipeline = Pipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
