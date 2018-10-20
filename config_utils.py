import json

CONFIG_MODEL_KEY = "Model"
CONFIG_MODEL_TYPE_SUBKEY = "Type"
CONFIG_MODEL_CKPTSAVEPATH_SUBKEY = "PathCKPT"
CONFIG_MODEL_CKPTFILENAME_SUBKEY = "NameCKPT"
CONFIG_TRAIN_KEY = "Train"
CONFIG_TRAIN_NOISE_SUBKEY = "Noise"
CONFIG_TRAIN_NOISE_COUNT_SUBSUBKEY = "Count"
CONFIG_TRAIN_NOISE_DIMENSIONS_SUBSUBKEY = "Dimensions"
CONFIG_TRAIN_NOISE_MEAN_SUBSUBKEY = "Mean"
CONFIG_TRAIN_NOISE_STDDEV_SUBSUBKEY = "STDDEV"
CONFIG_TRAIN_SAMPLES_SUBKEY = "Samples"
CONFIG_TRAIN_SAMPLES_COUNT_SUBSUBKEY = "Count"
CONFIG_TRAIN_SAMPLES_DIMENSIONS_SUBSUBKEY = "Dimensions"
CONFIG_TRAIN_SAMPLES_TYPE_SUBSUBKEY = "Type"
CONFIG_TRAIN_EPOCHS_SUBKEY = "Epochs"
CONFIG_TRAIN_EPOCHS_MAIN_SUBSUBKEY = "Main"
CONFIG_TRAIN_EPOCHS_GEN_SUBSUBKEY = "Gen"
CONFIG_TRAIN_EPOCHS_DISC_SUBSUBKEY = "Disc"
CONFIG_TRAIN_BATCHSIZE_SUBKEY = "BatchSize"
CONFIG_TRAIN_BATCHSIZE_GEN_SUBSUBKEY = "Gen"
CONFIG_TRAIN_BATCHSIZE_DISC_SUBSUBKEY = "Disc"
CONFIG_TRAIN_DEBUG_KEY = "Debug"
CONFIG_TRAIN_DEBUG_INTERVALS_SUBKEY = "Intervals"
CONFIG_TRAIN_DEBUG_INTERVALS_PRINTLOSSGEN_SUBSUBKEY = "PrintLossGen"
CONFIG_TRAIN_DEBUG_INTERVALS_PRINTLOSSDISC_SUBSUBKEY = "PrintLossDisc"
CONFIG_TRAIN_DEBUG_INTERVALS_SAVECKPT_SUBSUBKEY = "SaveCKPT"
CONFIG_TEST_KEY = "Test"
CONFIG_TEST_NOISE_SUBKEY = "Noise"
CONFIG_TEST_NOISE_COUNT_SUBSUBKEY = "Count"
CONFIG_TEST_OUTPUTTENSOR_SUBKEY = "OutputTensorName"
CONFIG_TEST_OUTPUTFOLDER_SUBKEY = "OutputFolder"

class ConfigLoader:
	def __init__(self, file_json):
		f = open(file_json)
		self._config = json.load(f)
		self.load_configs()
		f.close()
		
	def load_configs(self):
		self.model_type = self._config[CONFIG_MODEL_KEY][CONFIG_MODEL_TYPE_SUBKEY]
		self.ckpt_save_path = self._config[CONFIG_MODEL_KEY][CONFIG_MODEL_CKPTSAVEPATH_SUBKEY]
		self.ckpt_file_name = self._config[CONFIG_MODEL_KEY][CONFIG_MODEL_CKPTFILENAME_SUBKEY]
		self.noise_count = self._config[CONFIG_TRAIN_KEY][CONFIG_TRAIN_NOISE_SUBKEY][CONFIG_TRAIN_NOISE_COUNT_SUBSUBKEY]
		self.noise_dims = self._config[CONFIG_TRAIN_KEY][CONFIG_TRAIN_NOISE_SUBKEY][CONFIG_TRAIN_NOISE_DIMENSIONS_SUBSUBKEY]
		self.noise_mean = self._config[CONFIG_TRAIN_KEY][CONFIG_TRAIN_NOISE_SUBKEY][CONFIG_TRAIN_NOISE_MEAN_SUBSUBKEY]
		self.noise_stddev =  self._config[CONFIG_TRAIN_KEY][CONFIG_TRAIN_NOISE_SUBKEY][CONFIG_TRAIN_NOISE_STDDEV_SUBSUBKEY]
		self.samples_count = self._config[CONFIG_TRAIN_KEY][CONFIG_TRAIN_SAMPLES_SUBKEY][CONFIG_TRAIN_NOISE_COUNT_SUBSUBKEY]
		self.samples_dims = self._config[CONFIG_TRAIN_KEY][CONFIG_TRAIN_SAMPLES_SUBKEY][CONFIG_TRAIN_SAMPLES_DIMENSIONS_SUBSUBKEY]
		self.samples_type = self._config[CONFIG_TRAIN_KEY][CONFIG_TRAIN_SAMPLES_SUBKEY][CONFIG_TRAIN_SAMPLES_TYPE_SUBSUBKEY]
		self.num_epochs_main = self._config[CONFIG_TRAIN_KEY][CONFIG_TRAIN_EPOCHS_SUBKEY][CONFIG_TRAIN_EPOCHS_MAIN_SUBSUBKEY]
		self.num_epochs_gen = self._config[CONFIG_TRAIN_KEY][CONFIG_TRAIN_EPOCHS_SUBKEY][CONFIG_TRAIN_EPOCHS_GEN_SUBSUBKEY]
		self.num_epochs_disc = self._config[CONFIG_TRAIN_KEY][CONFIG_TRAIN_EPOCHS_SUBKEY][CONFIG_TRAIN_EPOCHS_DISC_SUBSUBKEY]
		self.batch_size_gen = self._config[CONFIG_TRAIN_KEY][CONFIG_TRAIN_BATCHSIZE_SUBKEY][CONFIG_TRAIN_BATCHSIZE_GEN_SUBSUBKEY]
		self.batch_size_disc = self._config[CONFIG_TRAIN_KEY][CONFIG_TRAIN_BATCHSIZE_SUBKEY][CONFIG_TRAIN_BATCHSIZE_DISC_SUBSUBKEY]
		self.print_interval_loss_gen = self._config[CONFIG_TRAIN_DEBUG_KEY][CONFIG_TRAIN_DEBUG_INTERVALS_SUBKEY][CONFIG_TRAIN_DEBUG_INTERVALS_PRINTLOSSGEN_SUBSUBKEY]
		self.print_interval_loss_disc = self._config[CONFIG_TRAIN_DEBUG_KEY][CONFIG_TRAIN_DEBUG_INTERVALS_SUBKEY][CONFIG_TRAIN_DEBUG_INTERVALS_PRINTLOSSDISC_SUBSUBKEY]
		self.save_interval_ckpt = self._config[CONFIG_TRAIN_DEBUG_KEY][CONFIG_TRAIN_DEBUG_INTERVALS_SUBKEY][CONFIG_TRAIN_DEBUG_INTERVALS_SAVECKPT_SUBSUBKEY]
		self.num_test_samples = self._config[CONFIG_TEST_KEY][CONFIG_TEST_NOISE_SUBKEY][CONFIG_TEST_NOISE_COUNT_SUBSUBKEY]
		self.output_tensor_name = self._config[CONFIG_TEST_KEY][CONFIG_TEST_OUTPUTTENSOR_SUBKEY]
		self.output_folder = self._config[CONFIG_TEST_KEY][CONFIG_TEST_OUTPUTFOLDER_SUBKEY]
		

