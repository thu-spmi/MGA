import logging, time, os

class _Config:
    def __init__(self):
        self._multiwoz_damd_init()

    def _multiwoz_damd_init(self):

        self.notes=''
        # file path setting
        self.data_path = './data/multi-woz-2.1-processed/'
        self.data_file = 'data_for_rl.json'
        self.dev_list = 'data/MultiWOZ_2.1/valListFile.txt'
        self.test_list = 'data/MultiWOZ_2.1/testListFile.txt'
        self.dbs = {
            'attraction': 'db/attraction_db_processed.json',
            'hospital': 'db/hospital_db_processed.json',
            'hotel': 'db/hotel_db_processed.json',
            'police': 'db/police_db_processed.json',
            'restaurant': 'db/restaurant_db_processed.json',
            'taxi': 'db/taxi_db_processed.json',
            'train': 'db/train_db_processed.json',
        }
        self.domain_file_path = 'data/multi-woz-2.1-processed/domain_files.json'
        self.slot_value_set_path = 'db/value_set_processed.json'
        self.exp_path = 'to be generated'
        self.log_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        
        # supervised training settings
        self.gpt_path = 'distilgpt2' # the pretrained model path
        self.lr = 1e-4 # learning rate
        self.warmup_steps = -1 # we use warm up ratio if warm up steps is -1
        self.warmup_ratio= 0.2 
        self.weight_decay = 0.0 
        self.gradient_accumulation_steps = 4
        self.batch_size = 8
        self.loss_reg=True # regularization for gradient accumulation
        self.gradient_checkpoint=False # use gradient checkpoint to accelerate training

        self.turn_level=True # turn-level training or session-level training
        self.model_act=True # add system act to the training sequence
        self.save_type='max_score'# evaluate the model on dev set with 'min_loss' or 'max_score'
        self.dataset=1 # 0 for multiwoz2.0, 1 for multiwoz2.1
        self.delex_as_damd = True # delexicalize as DAMD
        self.input_history=False # add the whole dialog history into the training sequence if train with turn-level 
        self.input_prev_resp=True # add the prev response into the training sequence if input_history is False
        self.fix_data=True # correct the dataset
        self.example_log=True # show some training sequence examples
        self.debugging=False # debugging mode

        # semi-supervised training settings
        self.spv_proportion=20 # the proportion (%) of supervised samples
        self.posterior_train=False # pretrain the posterior model (inference model) in VL
        self.divided_path='to be generated' # data file after the original data is divided into supervised and unsupervised parts
        # semi-VL training
        self.PrioriModel_path='to be filled' # the path of pretrained Prior model (generative model)
        self.PosteriorModel_path='to be filled' # the path of pretrained Posterior model (inference model)

        # evaluation settings
        self.eval_batch_size=32
        self.col_samples=False # collect wrong predictions samples
        self.use_existing_result=True # use the generated result file to evaluate
        self.result_file='validate_result.json' # the file name of generated result
        self.eval_load_path = 'to be generated' # the model to be evaluated
        self.eval_as_simpletod=True # evaluate like SimpleTOD
        # below are settings from UBAR, set all False for end-to-end evaluation
        self.use_true_prev_bspn = False
        self.use_true_prev_aspn = False
        self.use_true_db_pointer = False
        self.use_true_prev_resp = False
        self.use_true_curr_bspn = False
        self.use_true_curr_aspn = False

        # other training settings
        self.mode = 'train' # 'train'/'pretrain'/'test'/'semi_VL'
        self.cuda_device = [0]
        self.exp_no = '' # experiment name
        self.seed = 11
        self.save_log = True # tensorboard log
        self.evaluate_during_training = True # evaluate during training
        self.use_scheduler=True
        self.epoch_num = 50
        self.early_stop=False
        self.early_stop_count = 5
        self.only_target_loss=True # only calculate the loss on target context
        self.clip_grad=True # gradient clipping
        self.exp_domains = ['all'] # e.g. ['attraction', 'hotel'], ['except', 'attraction']
        self.init_eval=False # whether evaluate the model before training

       
        # old or useless settings from DAMD
        self.use_true_bspn_for_ctr_eval = False 
        self.use_true_domain_for_ctr_eval = True
        self.use_true_domain_for_ctr_train = True
        self.vocab_size = 3000
        self.enable_aspn = True
        self.enable_bspn = True
        self.bspn_mode = 'bspn' # 'bspn' or 'bsdx'
        self.enable_dspn = False # removed
        self.enable_dst = False
        # useless settings
        self.multi_acts_training = False
        self.same_eval_act_f1_as_hdsa = False
        
        
        

    def __str__(self):
        s = ''
        for k,v in self.__dict__.items():
            s += '{} : {}\n'.format(k,v)
        return s


    def _init_logging_handler(self, mode):
        stderr_handler = logging.StreamHandler()
        if not os.path.exists('./log'):
            os.mkdir('./log')
        if self.save_log and mode in ['semi_ST', 'semi_VL', 'semi_jsa', 'train', 'pretrain']:
            if self.dataset==0:
                file_handler = logging.FileHandler('./log/log_{}_{}_sd{}.txt'.format(mode, self.exp_no, self.seed))
            elif self.dataset==1:
                file_handler = logging.FileHandler('./log21/log_{}_{}_sd{}.txt'.format(mode, self.exp_no, self.seed))
        elif 'test' in mode and os.path.exists(self.eval_load_path):
            eval_log_path = os.path.join(self.eval_load_path, 'eval_log.json')
            # if os.path.exists(eval_log_path):
            #     os.remove(eval_log_path)
            file_handler = logging.FileHandler(eval_log_path)
            file_handler.setLevel(logging.INFO)
        else:
            pass
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()