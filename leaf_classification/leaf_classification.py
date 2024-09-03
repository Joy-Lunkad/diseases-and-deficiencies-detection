"""leaf_classification dataset."""

import tensorflow_datasets as tfds
import os
from pathlib import Path
from collections import Counter
import tensorflow as tf

# TODO(leaf_classification): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(leaf_classification): BibTeX citation
_CITATION = """
"""


class LeafClassification(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for leaf_classification dataset."""

  VERSION = tfds.core.Version('4.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Register into https://example.org/login to get the data. Place the `data.zip`
  file in the `manual_dir/`.
  """
    
  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(leaf_classification): Specifies the tfds.core.DatasetInfo object
    self.IM_SIZE = 600

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(self.IM_SIZE, self.IM_SIZE, 3)),
            'label': tfds.features.ClassLabel(num_classes=15),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    # Download source data
    '''
    manually download the zip file of the dataset and cp it into
    root/tensorflow_datasets/downloads/manual
    '''
    archive_path = dl_manager.manual_dir / 'full'

    # Extract the manually downloaded `data.zip`
    # extracted_path = dl_manager.extract(archive_path)

    # print(os.getcwd())
    # os.chdir('..')
    # extracted_path = Path('full')
    # print(os.getcwd())
    # Specify the splits
    
    path = archive_path / 'leaf_cropped_images'
    print(path)
    
    temp = Counter([os.path.basename(os.path.dirname(l)) for l in path.glob('*/*.*')])
    self.label_encoder = {k:v for (v,k) in enumerate(temp.keys())}

    return {
        'entire_dataset': self._generate_examples(
            path=path, split_size=[0, 6620],
        ),
        'train': self._generate_examples(
            path=path, split_size=[0, 5296],
        ),
        'val': self._generate_examples(
            path=path, split_size=[5296, 5958],
        ),
        'test': self._generate_examples(
            path=path, split_size=[5958, 6620],
        ),
    }

  def _generate_examples(self, path, split_size):
    """Yields examples."""
    # TODO(leaf_classification): Yields (key, example) tuples from the dataset  

    all_files = sorted([p for p in path.glob('*/*.*') if '.db' not in str(p) \
                 and '(1)' not in str(p) and 'resized' not in str(p)])
    
    start, end = split_size
    all_files = all_files[start:end]

    duplicate_key_checker = set()

    for img_path in all_files:
        # print('img_path: ', img_path)
        try:
            out_dir_path =  str(os.path.dirname(img_path)) 
            out_img_path = f'resized_{self.IM_SIZE}' + str(os.path.basename(img_path))
            out_path = os.path.join(out_dir_path, out_img_path)
            if not os.path.exists(out_path):
                im = tf.keras.utils.load_img(str(img_path))
                # print('org img: ',im.size)
                im = im.resize((self.IM_SIZE, self.IM_SIZE), resample=3)
                # print('resized img: ',im.size)
                # print('saving at: ', out_path)
                im.save(out_path, "JPEG")
            
            key = os.path.basename(out_path)
            if key not in duplicate_key_checker:
                yield key, {
                'image': out_path,
                'label': self.label_encoder[os.path.basename(os.path.dirname(out_path))],
                }   
            duplicate_key_checker.add(key)      
        except:
            print(f'broken example -> {img_path}')
            continue

