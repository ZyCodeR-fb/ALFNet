python main.py `
  --model MoGERNN `
  --input_features v --target_features v `
  --d_model 64 `
  --input_len 12 --pred_len 12 --look_back 0 `
  --dloader_name METRLA `
  --batch_size 64 `
  --loss mse `
  --data_path ./data/METR-LA/raw_data.pkl `
  --num_workers 8 `
  --patience 10 `
  --k_hop 1 `
  --unknown_nodes_path ./data/METR-LA/unknown_nodes.npy `
  --slide_step 12 `
  --scheduler None `
  --mean_expert --weight_expert --max_expert --min_expert --diffusion_expert `
  --stop_based val_mask `
  --seed 32 `
  --num_used_experts 3 `
  --test_for_async_lifecycle `
  --async_adaptor base `
  --async_num_events 4 `
  --async_commissioning_k 12 `
  --async_event_seed 0



python main.py `
  --model MoGERNN `
  --input_features v --target_features v `
  --d_model 64 `
  --input_len 12 --pred_len 12 --look_back 0 `
  --dloader_name METRLA `
  --batch_size 64 `
  --loss mse `
  --data_path ./data/METR-LA/raw_data.pkl `
  --num_workers 8 `
  --patience 10 `
  --k_hop 1 `
  --unknown_nodes_path ./data/METR-LA/unknown_nodes.npy `
  --slide_step 12 `
  --scheduler None `
  --mean_expert --weight_expert --max_expert --min_expert --diffusion_expert `
  --stop_based val_mask `
  --seed 32 `
  --num_used_experts 3 `
  --test_for_async_lifecycle `
  --async_adaptor st_ttc `
  --async_num_events 4 `
  --async_commissioning_k 12 `
  --async_event_seed 0

