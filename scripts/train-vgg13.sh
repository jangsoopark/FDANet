#!/bin/bash

project_root="$(dirname ${PWD})"
train=${project_root}/src/train.py

echo $train 

# T_0 = 5, T_mult = 2
python $train --config-name='config/vgg13/VGG13BN-FDAM-LEVIR-schedule-T_0-5-T_mult-2'
python $train --config-name='config/vgg13/VGG13BN-FDAM-WHU-schedule-T_0-5-T_mult-2'

# T_0 = 5, T_mult = 4
python $train --config-name='config/vgg13/VGG13BN-FDAM-LEVIR-schedule-T_0-5-T_mult-4'
python $train --config-name='config/vgg13/VGG13BN-FDAM-WHU-schedule-T_0-5-T_mult-4'

# T_0 = 10, T_mult = 2
python $train --config-name='config/vgg13/VGG13BN-FDAM-LEVIR-schedule-T_0-10-T_mult-2'
python $train --config-name='config/vgg13/VGG13BN-FDAM-WHU-schedule-T_0-10-T_mult-2'

# T_0 = 10, T_mult = 4
python $train --config-name='config/vgg13/VGG13BN-FDAM-LEVIR-schedule-T_0-10-T_mult-4'
python $train --config-name='config/vgg13/VGG13BN-FDAM-WHU-schedule-T_0-10-T_mult-4'

# T_0 = 15, T_mult = 2
python $train --config-name='config/vgg13/VGG13BN-FDAM-LEVIR-schedule-T_0-15-T_mult-4'
python $train --config-name='config/vgg13/VGG13BN-FDAM-WHU-schedule-T_0-15-T_mult-4'

# T_0 = 15, T_mult = 4
python $train --config-name='config/vgg13/VGG13BN-FDAM-LEVIR-schedule-T_0-15-T_mult-2'
python $train --config-name='config/vgg13/VGG13BN-FDAM-WHU-schedule-T_0-15-T_mult-2'