import os
import pathlib


current_path = pathlib.Path(__file__).parent.resolve()
stylegan3_repo = os.path.join(current_path, 'stylegan3_repo')


class StyleGAN3:

    def prepare_dataset(self, monet_dataset_source: str, monet_dataset_output: str, res: int = 256):

        dataset_tool = os.path.join(stylegan3_repo, 'dataset_tool.py')

        comm = f"python {dataset_tool} --source={monet_dataset_source} --dest={monet_dataset_output} " \
               f"--resolution={res}x{res}"

        os.system(comm)

    def train(self, dataset_path: str, resume_from: str, output_dir: str,
              batch_size: int = 32, batch_size_gpu: int = 16, gamma_value: int = 2, snapshot_count: int = 5,
              kimg: int = 100, workers: int = 2):

        train_module = os.path.join(stylegan3_repo, 'train.py')

        comm = f"python {train_module} --outdir='{output_dir}' --tick=2 --workers={workers} --mirror=1 " \
               f"--cfg=stylegan3-t --data={dataset_path} --gpus=1 --batch={batch_size} --batch-gpu={batch_size_gpu} " \
               f"--gamma={gamma_value} --snap={snapshot_count} --resume={resume_from} --kimg={kimg} " \
               "--freezed=10 --metrics=None "

        os.system(comm)

    def generate_images(self,  checkpoint_to_load: str, output_directory: str = 'stylegan3_output'):

        gen_module = os.path.join(stylegan3_repo, 'gen_images.py')

        comm = f"python {gen_module} --outdir={output_directory} --network={checkpoint_to_load} --noise-mode='random' " \
               f"--seeds=69-420"

        os.system(comm)
