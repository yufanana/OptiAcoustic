# Opti-Acoustic Sensor Fusion

## To-do

Python Simulation

- [ ] analyse effects of offsetting Qw
- [ ] pitch sonar and camera downwards
- [ ] test convering camera set up (rotate sonar only)

DAVE Simulation

- [ ] Add sonar as link to robot
- [ ] Teleop robot

MLE debugging takeaways

- Epipolar parallel camera: U matrix is only valid for tx != 0
- Take absolute value of epipolar value
- Follow the scale provided in the paper
- focal length, f, is needed in the p vector for epipolar calculation
- 
