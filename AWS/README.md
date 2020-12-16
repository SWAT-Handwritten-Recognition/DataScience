# Execution of the AWS Instance

All the process to include the DS Instance in AWS was based on the next YouTube video: [Tutorial 4- Deployment Of ML Models In AWS EC2 Instance](https://www.youtube.com/watch?v=oOqqwYI60FI&list=PLZoTAELRMXVOAvUbePX1lTdxQR8EY35Z1&index=4&ab_channel=KrishNaik).

**Some considerations:**
* The software explained in the video is only for Windows. The developers of this project work on linux, so they had to find software to make the connections.
* The software used to configure the folder in AWS was not WinSCP, but Filezilla.
* The PEM file downloaded from AWS can be used to configure Filezilla.
* Due to PuTTY does not work on linux, it was used *electerm* as an alternative. `sudo snap install electerm`.
