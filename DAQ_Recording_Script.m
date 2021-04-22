clear all % clears the workspace
close all % closes all open figures

filename = 'Test_Data'; % file name for saving data in a Matlab .mat file
                        % Name 'Calibration Data' when performing
                        % calibraton. Name 'Test Data' when collecting test
                        % data.

fs=1000;        % sampling frequency (Hz)
record_time=30;  % amount of time to record data (sec)

% Create a DAQ session within Matlab to look for the NI (National
% Instruments) DAQ

s=daq.createSession('ni');

% Tell Matlab to use one analog input channel from device number 'devX'
% NOTE: you'll have to update the device number depending on what is
% assigned when you connect the DAQ to the computer

[ch,idx]=s.addAnalogInputChannel('dev1','ai0','Voltage');

% specify the voltage range of the channel. If your range is too small 
% your data peaks will be "clipped" at the maximum/minimum values you 
% specify

ch(1).Range=[-10 10];

% set the sampling frequency (fs) and duration of the recording
% (record_time)

s.Rate=fs;
s.DurationInSeconds=record_time;

% We must first create a listener to allow us to look at the data 
% mid-stream and make sure there is data in the first place. In order
% to do this, we will first need to add the following "listener" 

listen=s.addlistener('DataAvailable',@(s,event) plot(event.TimeStamps,event.Data));

% start data capture, where "data" is the recorded voltage signal and "t" is the time in
% seconds
[data,t]=s.startForeground();

% plot your final data vs. time
plot(t,data); xlabel('Time (s)'); ylabel('Voltage (V)');

% save a copy of the plot 
saveas(gcf,filename,'fig');

% save your data as a .mat file so you can load it into Matlab later using
% the load() command
save(filename);
