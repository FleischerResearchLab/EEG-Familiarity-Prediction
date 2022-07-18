% Performs various preprocessing by trial.
% Changes eeglab based 3D matrix into cell arrays containing each trial.
% Consists of filtering, re-referencing and downsampling
% INPUTS
% == required ==
% input_struct: eeglab based structure
% == optional(could be set as empty) ==
% reref: 0/1(default: 0)
% reref_t: re-reference length starting from -xmin sec (default: abs(xmin))
% freq_bands: [w1 w2 w3 w4...](default: []), if empty no filtering
% f_order: filter order of FIR filter
% downrate:[p q](default: [1 1]) where p/q: downsampling ratio
% t_interval:[start_t end_t](default: [xmin xmax]) start and end of
% interval in seconds
% 
% OUTPUTS
% output_data: Cell array containing cells of individual trials
% output_struct: eeglab based structure containing preprocessed data and
% param

function [ output_data output_struct fir_filt] = preprocess_by_trial( input_struct, reref, reref_t, freq_bands, f_order, downrate, t_interval)

% get params from structure

xmin = input_struct.xmin;
xmax = input_struct.xmax;
srate = input_struct.srate;
input_data = double(input_struct.data);

% get params from inputs
if(isempty(reref))
    reref = 0;
end

if(isempty(reref_t))
    reref_t = abs(xmin);
end

if(isempty(freq_bands))
    freq_bands = 0;
end

if(isempty(downrate))
    p = 1;
    q = 1;
elseif(length(downrate) == 2)
    p = downrate(1);
    q = downrate(2);
else
    display('error in downrate');
    return
end

if(isempty(t_interval))
    start_t = xmin;
    end_t = xmax;
    start_s = 1;
    end_s = length(input_struct.times);
elseif(length(t_interval) == 2)
    start_t = t_interval(1);
    end_t = t_interval(2);
    start_s = floor( (start_t - xmin)*srate)+1;
    end_s = floor( (end_t - xmin)*srate);
else
    display('error in t_interval');
    return
end

%%%%% PREPROCESSING %%%%%
% initialization
[ num_of_channels, seq_length, num_of_trials ] = size(input_data);

output_data = cell(1);

if( freq_bands ~= 0 )
    % define FIR filter
    fir_filt = fir1(f_order, freq_bands/(srate/2));
    %[b a] = yulewalk(f_order,[ 0 freq_bands/(srate/2) freq_bands(2)/(srate/2) 1], [0 1 1 0 0]);
    % uncomment to plot fir filter
    %freqz(fir_filt,1,srate);
end

str_hist = '';

for trial = 1:num_of_trials
    
    temp_data1 = input_data(:,:,trial);
    
    % if reref == 0 no re-referencing
    if( reref ~= 0 )
        reref_max = floor(abs(reref_t)*srate);        
        ref_mean = mean(temp_data1(:, 1:reref_max), 2);
        temp_data1 = temp_data1 - repmat(ref_mean, size(temp_data1(1,:)));
    end
    
    % select pre-defined interval
    temp_data1 = temp_data1(:, start_s:end_s);
    
    [ tmp1, seq_length, tmp2 ] = size(temp_data1);
   
    % conduct filtering after interval selection
    % if freq_bands == 0 no filtering
    if( freq_bands ~= 0 )
        temp_data1 = (filtfilt(fir_filt, 1, temp_data1')');
        %temp_data1 = (filtfilt(b, a, temp_data1')');
    end

    % if p == q no dowsampling, if p>q then error
    if(p > q)
        display('Error while downsampling, no downsampling performed');
    elseif(p < q)
        for i = 1:num_of_channels
            temp_data2(i,:) = resample([0, temp_data1(i, 2:seq_length)], p, q);
        end
    else
        temp_data2 = temp_data1;
    end
    
    output_data{trial}(:,:) = temp_data2;

end

if(freq_bands ~=0 )
    str_hist = [str_hist, [' filtered with filter ', num2str(freq_bands)]];
end
if(reref ~= 0 )
    str_hist = [str_hist, [' re-reference with reference from ', num2str(xmin), ' to ' num2str(xmin+reref_t)]];
end
if( p < q)
    str_hist = [str_hist, [' dowsampled to freq ', num2str(srate*p/q)]];
end


output_struct = input_struct;
output_struct.data = single(cell_to_mat(output_data));
output_struct.srate = srate*p/q;
output_struct.xmin = start_t;
output_struct.xmax = end_t;
output_struct.times = input_struct.times(start_s:end_s);
output_struct.history = [input_struct.history, str_hist];




