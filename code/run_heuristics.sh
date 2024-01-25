for heuritstic_id in 1 2 3 4 5
do
sed -i "s/^heuristic_dict_id = .*/heuristic_dict_id = ${heuritstic_id}/" ./data_file.py;
python3 main.py
python3 compile_data.py
done
