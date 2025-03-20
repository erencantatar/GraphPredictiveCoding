
- Discrim.
   - Zwol: GOOD
  python train.py     --model_type IPC   --task classification  --dataset_transform normalize_mnist_mean_std     --graph_type single_hidden_layer --discriminative_hidden_layers 50,30 --generative_hidden_layers 0     --update_rules vanZwol_AMB  --delta_w_selection all     --weight_init "normal 0 0.005"     --use_grokfast False     --optimizer 1     --remove_sens_2_sens True     --remove_sens_2_sup True     --set_abs_small_w_2_zero False     --mode experimenting     --use_wandb disabled     --tags PC_vs_IPC     --use_bias False     --set_abs_small_w_2_zero False     --normalize_msg False     --numbers_list 0,1,2,3,4,5,6,7,8,9     --N all     --supervision_label_val 1     --num_internal_nodes 1000     --T_train 10 --T_test 10  --lr_values 0.5     --lr_weights 0.00001     --activation_func relu     --epochs 20     --batch_size 100     --seed 2 


   - MP


- Generative; overfit fast; 
   - Zwol: GOOD ; bit weird; can train to much"??!
 python train.py --model_type IPC --dataset_transform normalize_mnist_mean_std --graph_type single_hidden_layer --discriminative_hidden_layers 0 --generative_hidden_layers 200,200 --task generation --update_rules vanZwol_AMB --delta_w_selection all --weight_init "fixed 0.0001 0.0001" --use_grokfast False --optimizer 1 --remove_sens_2_sens True --remove_sens_2_sup False --set_abs_small_w_2_zero False --mode experimenting --use_wandb disabled --tags PC_vs_IPC --use_bias False --set_abs_small_w_2_zero False --normalize_msg False --numbers_list 0,1,2,3,4,5,6,7,8,9 --N all --supervision_label_val 1 --num_internal_nodes 100 --T_train 10 --T_test 15 --lr_values 0.5 --lr_weights 0.000001 --activation_func tanh --epochs 20 --batch_size 50 --seed 20 --break_num_train 20


   - MP: GOOD
python train.py --model_type IPC --dataset_transform normalize_mnist_mean_std --graph_type single_hidden_layer --discriminative_hidden_layers 0 --generative_hidden_layers 200,200 --task generation --update_rules MP_AMB --delta_w_selection all --weight_init "fixed 0.0001 0.0001" --use_grokfast False --optimizer 1 --remove_sens_2_sens True --remove_sens_2_sup False --set_abs_small_w_2_zero False --mode experimenting --use_wandb online --tags PC_vs_IPC --use_bias False --set_abs_small_w_2_zero False --normalize_msg False --numbers_list 0,1,2,3,4,5,6,7,8,9 --N all --supervision_label_val 1 --num_internal_nodes 100 --T_train 10 --T_test 15 --lr_values 0.5 --lr_weights 0.000001 --activation_func tanh --epochs 20 --batch_size 50 --seed 20 --break_num_train 20


- Fully Conn:
   - Zwol
   - MP
