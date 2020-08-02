#!/bin/bash
echo "Negative to positive"
declare -a arr=("my goodness it was so gross ." 
"the steak was rough and bad ." 
"it really feels like a total lack of effort , honesty and professionalism .")

## now loop through the above array
for i in "${arr[@]}"
do    
    python code/style_transfer.py \
    --predict "$i" \
    --model_path model_saves/model_0.001_100_Apr-02-2020_08-34-58/64_epochs \
    --target_sentiment 0 \
    --vocab tmp/yelp.vocab
   # or do whatever with individual element of the array
done

echo "Positive to negative"
declare -a arr2=("i love the ladies here !" 
"came here with my wife and her grandmother !" 
"the breakfast was the best and the women helping with the breakfast were amazing !")

## now loop through the above array
for i in "${arr2[@]}"
do    
    python code/style_transfer.py \
    --predict "$i" \
    --model_path model_saves/model_0.001_100_Apr-02-2020_08-34-58/64_epochs \
    --target_sentiment 1 \
    --vocab tmp/yelp.vocab
   # or do whatever with individual element of the array
done
