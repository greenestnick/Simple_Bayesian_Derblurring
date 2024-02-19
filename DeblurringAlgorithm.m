%   Deblurring Algorithm with Known Gaussian Kernal
%   
%   Use: Make sure the matlab file is located in the same folder as the images, or extend the imread string to include the path
%           You must type the image name in the imread function to select the image
%
% =================================================================================================================================

clear; clc;

cleanImg = rgb2gray(imread(".\TestImages\Bird.jpg"));
cleanImg = imresize(cleanImg, 0.05);
cleanImg = double(cleanImg);

blurSigma = 2;
kernalSize = 2*(2*round(blurSigma)) + 1; %Must be odd, Matlab by default will use 2*(2*sigma) + 1 which is 4-sigma made odd
blurImg = imgaussfilt(cleanImg, blurSigma, "FilterSize", kernalSize);

guessImg = blurImg;
sampleMeans = zeros(size(blurImg));

MAX_ITER = 1e5;
iterErr = ones([1 MAX_ITER]);
step = 20 * ones(size(blurImg));

fprintf("Iteration: %05d\tRMS: %06.3f\n", i, RMS(cleanImg, guessImg));

for i = 1:MAX_ITER
    %Computing the posterior for the previous guess
    postOrig = PostAcceptanceRatio(blurImg, guessImg, blurSigma);
    postOrig(postOrig(:,:) < 1e-20) = 1e-20; %limiting the min to be above 0


    %Randomly modifying the guess to produce the next image and finding it's posterior 
    next = step .* randn(size(guessImg)) + guessImg;
    nextBlur = imgaussfilt(next, blurSigma, "FilterSize", kernalSize);
    nextPost = PostAcceptanceRatio(blurImg, next, blurSigma);


    %Getting the Acceptance Ratio
    AR = min(nextPost ./ postOrig, 1);

    %Keeping some of the guess pixels randomly 
    filter = rand(size(guessImg)) <= AR;
    compFilter = ones(size(filter)) - filter;
    guessImg = next.* filter + guessImg.*compFilter;


    %Sample average of the MCMC
    sampleMeans = (sampleMeans.*(i - 1) + guessImg) ./i;

    %Computing the RMS and checking when to stop
    iterErr(i) = RMS(cleanImg, sampleMeans);
    if(i > 2000)
        if (abs(iterErr(i - 1000) - iterErr(i)) < 0.02)
            fprintf("Ending loop at %d\n", i);
            fprintf("RSM Error = %f\n", iterErr(i));
            break;
        end
        
    end


    if(mod(i, 1000) == 0)
        fprintf("Iteration: %05d\tRMS: %06.3f\n", i, iterErr(i));
    end
    
end

imshow(sampleMeans./255);

%Simple gaussian likelihood returning the posterior
function post = PostAcceptanceRatio(blurImg, guessImg, blurSigma)
    kernalSize = 2*(2*blurSigma) + 1;
    gaussSigma = 1;

    blurGuess = imgaussfilt(guessImg, blurSigma, "FilterSize", kernalSize);
   
    likelihood = exp(-(blurImg - blurGuess).^2 ./(2*gaussSigma^2))./sqrt(2*pi*gaussSigma^2);
    prior = GaussianNeighborhood(guessImg);

    post = likelihood.* prior;
end

%Prior Probability based on the neighborhood of a pixel
function priorImg = GaussianNeighborhood(img)
    gaussSigma = 100;

    kernal = [0 1 0; 1 0 1; 0 1 0];


    convd = conv2(img, kernal, 'same');
    convd2 = conv2(img.^2, kernal, 'same');
    
    priorImg = 4.*img.^2 - 2.*img.*convd + convd2;
    priorImg =  exp(-(priorImg) ./(2*gaussSigma^2))./sqrt(2*pi*gaussSigma^2);
end


function err = RMS(A, B)
    err = sqrt(sum((A-B).^2, 'all')./ prod(size(A)));
end

