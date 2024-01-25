sed -i "s/^heuristic_dict_id = .*/heuristic_dict_id = 0/" ./data_file.py;
python3 main.py --train;
python3 merge_replay_buffer.py;
python3 train_on_merged_replay_buffer.py;
python3 main.py --test 100000;
python3 compile_data.py;
