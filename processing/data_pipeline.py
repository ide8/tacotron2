CODE_PATH = '/workspace/code/gst'
OUTPUT_DIRECTORY = '/workspace/training_data'

SR = 22050
TOP_DB = 40


DATA_CONFIG = {
        'pipeline': {
            'clean': True,
            'form_train_val': True
        },
        'data': [
            {
                'path': '/workspace/data/linda_johnson',
                'speaker_name': 'linda_johnson',
                'speaker_id': 0,
                'pipeline': {
                    'cleaning': {
                        'trim': True
                    }
                }
            },
            {
                'path': '/workspace/data/gcp/samantha_default',
                'speaker_name': 'samantha_default',
                'speaker_id': 1,
                'pipeline': {
                    'cleaning': {
                        'trim': True
                    }
                }
            },
            {
                'path': '/workspace/data/scarjo_her',
                'speaker_name': 'scarjo_her',
                'speaker_id': 1,
                'pipeline': {
                    'cleaning': {
                        'trim': True
                    }
                }
            },
            {
                'path': '/workspace/data/scarjo_the_dive_descript_grouped',
                'speaker_name': 'scarjo_the_dive_descript_grouped',
                'speaker_id': 1,
                'pipeline': {
                    'cleaning': {
                        'trim': True
                    }
                }
            },
            {
                'path': '/workspace/data/scarjo_the_dive_descript_ungrouped',
                'speaker_name': 'scarjo_the_dive_descript_ungrouped',
                'speaker_id': 1,
                'pipeline': {
                    'cleaning': {
                        'trim': True
                    }
                }
            },
            {
                'path': '/workspace/data/aws/dataset/blizzard_2013',
                'speaker_name': 'blizzard_2013',
                'speaker_id': 2,
                'pipeline': {
                    'cleaning': {
                        'trim': True
                    }
                }
            },
            {
                'path': '/workspace/data/aws/dataset/en_US/by_book/female/judy_bieber',
                'speaker_name': 'judy_bieber',
                'speaker_id': 3,
                'pipeline': {
                    'cleaning': {
                        'trim': True
                    }
                }
            },
            {
                'path': '/workspace/data/aws/dataset/en_US/by_book/female/mary_ann',
                'speaker_name': 'mary_ann',
                'speaker_id': 4,
                'pipeline': {
                    'cleaning': {
                        'trim': True
                    }
                }
            },
            {
                'path': '/workspace/data/aws/dataset/en_UK/by_book/female/elizabeth_klett',
                'speaker_name': 'elizabeth_klett',
                'speaker_id': 5,
                'pipeline': {
                    'cleaning': {
                        'trim': True
                    }
                }
            },
            {
                'path': '/workspace/data/aws/dataset/en_US/by_book/male/elliot_miller',
                'speaker_name': 'elliot_miller',
                'speaker_id': 6,
                'pipeline': {
                    'cleaning': {
                        'trim': True
                    }
                }
            },
        ]
    }

