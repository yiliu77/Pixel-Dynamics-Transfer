## Pixel-Level and Dynamics-Level Transfer Learning for RL

#### Policy
- SAC
  - SAC maximizes the entropy of the policy along with the reward of the policy, balancing exploration and exploitation
  - Very stable

#### Three discrminiators are used
- Pixel Discriminator
  - Encourages embeddings to be close for either environemnts
- State-Action Discriminator
  - Encourages actions to be similar for the two environments
- State-Action-Next State Discriminator
  - Encourages dynammics to be similar for the two environments

#### Environments
- Two Separate Cheetah Environments
  - Mujoco Cheetah Environments
  - Different shadings and coloring with different patterns on the floors for the two environments
  - Different dynamics including different gravity and a broken joint for one of the cheetahs
 
#### Results
Naive SAC trained on the two environments could not learn the new policy. However, utilizing the three discriminators reached maximum reward. 
