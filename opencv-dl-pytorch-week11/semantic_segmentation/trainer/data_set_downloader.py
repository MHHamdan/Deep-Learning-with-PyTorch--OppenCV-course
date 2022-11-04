import os
import errno

from torchvision.datasets.utils import download_url, extract_archive


class DataSetDownloader:
    def __init__(self, root_dir, dataset_title, download=False):
        """
            Parameters:
                root_dir: root directory of data set
                dataset_title: title of dataset for download
                download: flag, True - data set should be downloaded, else the value is False
        """
        self.root_dir = root_dir
        self.dataset_title = dataset_title
        self.download = download
        self.full_root_dir = os.path.join(self.root_dir, self.dataset_title)

        if download:
            self.download_dataset()

        if not os.path.exists(self.full_root_dir):
            raise RuntimeError('Data set was not found, please, use download=True to download it')

    def download_dataset(self):
        """ Download dataset if it does not exist in the specified directory """
        # pylint: disable=R1721
        if os.path.exists(self.full_root_dir) and len([elem for elem in os.scandir(self.full_root_dir)]) >= 2:
            print("The data set has been already downloaded")
            return

        # create root dir
        try:
            os.makedirs(self.full_root_dir)
        except OSError as error_obj:
            if error_obj.errno != errno.EEXIST:
                raise

        dataset_url = self._dataset_mapper()

        archive_name = dataset_url.rpartition('/')[2]
        self._get_dataset(dataset_url, filename=archive_name)

    def _dataset_mapper(self):
        """ Verifies data set title and returns url of data set"""

        dataset_map = {
            'tiny-imagenet-200': 'http://cs231n.stanford.edu/tiny-imagenet-200.zip',
            'PennFudanPed': 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
        }

        if self.dataset_title in dataset_map:
            return dataset_map[self.dataset_title]

        raise KeyError(
            '{} data set was not found. Check its title or choose another one from the list: {}'.format(
                self.dataset_title, list(dataset_map.keys())
            )
        )

    def _get_dataset(self, url, filename=None, remove_finished=True):
        self.full_root_dir = os.path.expanduser(self.full_root_dir)

        if not filename:
            filename = os.path.basename(url)

        print('Downloading {}...'.format(self.dataset_title))
        download_url(url, self.full_root_dir, filename)

        archive = os.path.join(self.full_root_dir, filename)
        print("Extracting {} to {}".format(archive, self.root_dir))
        extract_archive(archive, self.root_dir, remove_finished)
        print("Done!")
