mkdir data/big_bench/
cd data/big_bench/

proxy_prefix="https://ghproxy.com/"
repo_prefix="https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/"
single_tasks=("ruin_names" "temporal_sequences" "emoji_movie" "code_line_description" "penguins_in_a_table" "date_understanding")
conceptual_combinations_subtasks=("contradictions" "emergent_properties" "fanciful_fictional_combinations" "homonyms" "invented_words" "surprising_uncommon_combinations")
strange_stories_subtasks=("boolean" "multiple_choice")
symbol_interpretation_subtasks=("adversarial" "emoji_agnostic" "name_agnostic" "plain" "tricky")
tracking_shuffled_objects_subtasks=("three_objects" "five_objects" "seven_objects")
logical_deduction_subtasks=("three_objects" "five_objects" "seven_objects")


for task in "${single_tasks[@]}"
do
    wget -O "${task}.json" "${proxy_prefix}${repo_prefix}${task}/task.json"
done

for subtask in "${conceptual_combinations_subtasks[@]}"
do
    wget -O "conceptual_combinations_${subtask}.json" "${proxy_prefix}${repo_prefix}conceptual_combinations/${subtask}/task.json"
done

for subtask in "${strange_stories_subtasks[@]}"
do
    wget -O "strange_stories_${subtask}.json" "${proxy_prefix}${repo_prefix}strange_stories/${subtask}/task.json"
    
done

for subtask in "${symbol_interpretation_subtasks[@]}"
do
    wget -O "symbol_interpretation_${subtask}.json" "${proxy_prefix}${repo_prefix}symbol_interpretation/${subtask}/task.json"
done

for subtask in "${tracking_shuffled_objects_subtasks[@]}"
do
    wget -O "tracking_shuffled_objects_${subtask}.json" "${proxy_prefix}${repo_prefix}tracking_shuffled_objects/${subtask}/task.json"
done

for subtask in "${logical_deduction_subtasks[@]}"
do
    wget -O "logical_deduction_${subtask}.json" "${proxy_prefix}${repo_prefix}logical_deduction/${subtask}/task.json"
done
