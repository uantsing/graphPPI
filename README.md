Execute the command within the project directory to train model and do validation:

`python model_train.py --ppi_path ppi_data_Demo.txt --split random --p_feat_matrix ./protein_info/x_list_7.pt --p_adj_matrix ./protein_info/edge_list_12.npy --save_path ./result_save  --epoch_num 100`