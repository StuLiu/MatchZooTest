'''
--------------------------------------------------------
@File    :   test.py    
@Contact :   1183862787@qq.com
@License :   (C)Copyright 2017-2018, CS, WHU

@Modify Time : 2019/5/19 20:02     
@Author      : Liu Wang    
@Version     : 1.0   
@Desciption  : None
--------------------------------------------------------  
'''
import numpy as np
def load_data(file_name):
	result = list()
	with open(file_name, 'r', encoding='utf-8') as file:
		line_list = file.readlines()
		for line in line_list:
			result.append(line.split('\t'))
		result = np.array(result)
		X = result[:, 1:]
		Y = result[:, 0]
		return X, Y

train_data = load_data('raw_data/msr_paraphrase_train.txt')
test_data = load_data('raw_data/msr_paraphrase_test.txt')

import matchzoo.models.match_pyramid as mp
model = mp.MatchPyramid()
model.params['embedding_output_dim'] = 300
model.params['embedding_output_dim'] = 300
model.params['num_blocks'] = 2
model.params['kernel_count'] = [16, 32]
model.params['kernel_size'] = [[3, 3], [3, 3]]
model.params['dpool_size'] = [3, 10]
model.guess_and_fill_missing_params(verbose=0)
model.build()
model.compile()
# x=typing.Union[np.ndarray, typing.List[np.ndarray], dict]
model.fit(x=train_data[0],
	y=train_data[1],
	batch_size = 128,
	epochs=1000
)