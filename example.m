clear
close all
rng("default")
load('simulations_gauss.mat')
window=128;
overlap=100;
nfft=512;
fs=25000;
J=2;
signal = macierz_syg_complete(:,5);
kurtosis_original = kurtosis(signal);
[S,f,t] = spectrogram(signal,window,overlap,nfft,fs);
time=(1:fs)/fs;
r=10;
[W, H, err] = spanonmf(abs(S), J, 'approximationrank', r, 'numsamples', 1e4);
W = full(W);


best_kurtosis = 0;
best_filter = 0;
best_filtered = 0;

for j=1:J
    filtered = filter_signal(signal, full(W(:,j)));
    kurt = kurtosis(filtered);
    if kurt > best_kurtosis
        best_kurtosis = kurt;
        best_filter = W(:,j);
        best_filtered = filtered;
    end
end


figure
plot(time, signal)
title("Original signal, kurtosis: ", kurtosis_original)
xlabel("Time [s]")
ylabel("Amplitude")

figure
plot(best_filter,f)
title("Obtained best filter")
xlabel("Amplitude")
ylabel("Frequency [Hz]")

figure
plot(time, best_filtered)
title("Filtered signal, kurtosis: ", best_kurtosis)
xlabel("Time [s]")
ylabel("Amplitude")

