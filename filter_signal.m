function [filtered_signal]=filter_signal(signal,selector)

x=signal;
SK=selector;
SK=SK(:)';
SK=[SK,fliplr(SK(2:end-1))];
b = fftshift(real(ifft(SK))); 
xf = fftfilt(b,x);
filtered_signal=xf;