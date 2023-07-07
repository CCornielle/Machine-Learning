%% Jacomelo train
clear all
close all
clear
clc
%%%%%%%%%%%%%%%%%%%%%Se prepara la inicialización%%%%%%%%%%%%%
im_jaco = (imread('train_jacomelo.jpeg')); % Se cargan los datos para la función, imagen e umbral
um=0.3; % El umbral dependerá de la iluminación de nuestra imagen al igual que la definición de los objetos

[image,im_reshap_bground,propied_jaco,A,contras_jacome,mean_jacome,std_jacome]=segmenta(im_jaco,um); % Se llama a la función introduciendo los datos previamente identificados
%De esta función lo mas importante a obtener es contras que es una matriz
%que contará con la varianza de cada una de las habichuelas

%prob_jacome = [contras_jacome ; prob_j]'

%% Pinta train

%%%%%%%%%%%%%%%%%%%%%Se prepara la inicialización%%%%%%%%%%%%%
im_pinta = (imread('train_pinta.jpeg')); % Se cargan los datos para la función, imagen e umbral
um=0.3; % El umbral dependerá de la iluminación de nuestra imagen al igual que la definición de los objetos

[image_pinta,im_reshap_bground_pinta,propied_pinta,A_pinta,contras_pinta,mean_pinta,std_pinta]=segmenta(im_pinta,um); % Se llama a la función introduciendo los datos previamente identificados
%De esta función lo mas importante a obtener es contras que es una matriz
%que contará con la varianza de cada una de las habichuelas

%prob_p = normpdf(contras_pinta,mean_pinta,std_pinta);
%prob_p = normpdf(contras_jacome,mean_pinta,std_pinta)
%prob_pinta = [contras_pinta ; prob_p]';
% sumar los dos array resultante y divir entre el que me dan

%% Test

%%%%%%%%%%%%%%%%%%%%%Se prepara la inicialización%%%%%%%%%%%%%
im_test = (imread('test_habichuelas.jpeg')); % Se cargan los datos para la función, imagen e umbral
um=0.2; % El umbral dependerá de la iluminación de nuestra imagen al igual que la definición de los objetos

[image_test,im_reshap_bground_test,propied_test,A_test,contras_test]=segmenta(im_test,um); % Se llama a la función introduciendo los datos previamente identificados
%De esta función lo mas importante a obtener es contras que es una matriz
%que contará con la varianza de cada una de las habichuelas

%%%%%% Con lo aprendido en con el training set ( La media y la varianza )
%%%%%% Se determina la probabilidad aunque no seria mas bien como una
%%%%%% estimacion de si es jacomelo o gira porque tal y como resulta su
%%%%%% sumatoria no da un resultado de 1

%% histogramas 
% Se saca los histogramas de la data que se va a utilizar para conocer a
% que distribucion se ajustan mejor, según los histogramas graficados la
% data tanto de jacomelo y pinta
figure()
subplot(2,1,1)
histfit(contras_jacome,10,'Normal')
title("Histograma de la varianza de las Jacomelo")
subplot(2,1,2)
histfit(contras_pinta,10,'Normal')
title("Histograma de la varianza de las Pinta")

%% algoritmo probando train jacomelo 

prob_j = normpdf(contras_test,mean_jacome,std_jacome);

prob_p = normpdf(contras_test,mean_pinta,std_pinta);
result = ones(1,100);

label_real= ones(1,100);

figure, imshow(im_jaco)
i=1;

prob_j = normpdf(contras_jacome,mean_jacome,std_jacome);

prob_p = normpdf(contras_jacome,mean_pinta,std_pinta);
result = ones(1,100);

for n=1:size(propied_jaco,1) %En este for del tamaño como cantidad de objetos a los que la funcion regionprops le saca las
%propiedades, se utiliza para poder detectar los objetos en la imagen a
%partir de ahi se asi la estimacion de su es jacomelo o pinta, se encierra
%en un rectangulo si es jacomelo y en otro si es pinta
    
    if propied_jaco(n).Area>100
            if prob_j(i)>prob_p(i)
            result(i)= 1;
            %Si se cumple se estima como jacomelo 
            rectangle('Position',propied_jaco(n).BoundingBox,'EdgeColor','g','LineWidth',2);
            i=i+1;
            else
            result(i)= 0; %sino significa que se estima coma gira
            rectangle('Position',propied_jaco(n).BoundingBox,'EdgeColor','r','LineWidth',2);
            i=i+1;
            end
    end
end
tasa_aciertos_jaco = sum((result == label_real))/length(contras_jacome)

%% algoritmo probando train Pinta

prob_j = normpdf(contras_pinta,mean_jacome,std_jacome);

prob_p = normpdf(contras_pinta,mean_pinta,std_pinta);
result = ones(1,100);

label_real= ones(1,100);

figure, imshow(im_pinta)
i=1;

for n=1:size(propied_pinta,1) %En este for del tamaño como cantidad de objetos a los que la funcion regionprops le saca las
%propiedades, se utiliza para poder detectar los objetos en la imagen a
%partir de ahi se asi la estimacion de su es jacomelo o pinta, se encierra
%en un rectangulo si es jacomelo y en otro si es pinta
    
    if propied_pinta(n).Area>100
            if prob_j(i)>prob_p(i)
            result(i)= 0;
            %Si se cumple se estima como jacomelo 
            rectangle('Position',propied_pinta(n).BoundingBox,'EdgeColor','g','LineWidth',2);
            i=i+1;
            else
            result(i)= 1; %sino significa que se estima coma gira
            rectangle('Position',propied_pinta(n).BoundingBox,'EdgeColor','r','LineWidth',2);
            i=i+1;
            end
    end
end
tasa_aciertos_pinta = sum((result == label_real))/length(contras_jacome)


%% Test estimacion
figure, imshow(im_test)
i=1;


prob_j = normpdf(contras_test,mean_jacome,std_jacome);

prob_p = normpdf(contras_test,mean_pinta,std_pinta);
result = ones(1,100);

label_real= [0 1 0 1 1 0 1 1 0 1 1 0 1 1 0 0 0 1 0 0 1 1 1 0 1 0 0 0 1 0  1 0 0 0 0 0 1 0 0 0 1 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0 1 1 0 1 1 0 0 0 1 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 1 0 1 0 1 0 1 1 1];

for n=1:size(propied_test,1) %En este for del tamaño como cantidad de objetos a los que la funcion regionprops le saca las
%propiedades, se utiliza para poder detectar los objetos en la imagen a
%partir de ahi se asi la estimacion de su es jacomelo o pinta, se encierra
%en un rectangulo si es jacomelo y en otro si es pinta
    
    if propied_test(n).Area>100
            if prob_j(i)>prob_p(i)
            result(i)= 1;
            %Si se cumple se estima como jacomelo 
            rectangle('Position',propied_test(n).BoundingBox,'EdgeColor','g','LineWidth',2);
            i=i+1;
            else
            result(i)= 0; %sino significa que se estima coma gira
            rectangle('Position',propied_test(n).BoundingBox,'EdgeColor','r','LineWidth',2);
            i=i+1;
            end
    end
    
   
end

tasa_aciertos = sum((result == label_real))/length(contras_test)



