%% Jacomelo train
clear all
close all
clear
clc
%%%%%%%%%%%%%%%%%%%%%Se prepara la inicialización%%%%%%%%%%%%%
im_jaco = (imread('train_jacomelo.jpeg')); % Se cargan los datos para la función, imagen e umbral
um=0.3; % El umbral dependerá de la iluminación de nuestra imagen al igual que la definición de los objetos

[image,im_reshap_bground,propied_jaco,A,contras_jacome,mean_jacome,std_jacome,medias_jacome,mean_medias_jacome,std_medias_jacome]=segmenta(im_jaco,um); % Se llama a la función introduciendo los datos previamente identificados
%De esta función lo mas importante a obtener es contras que es una matriz
%que contará con la varianza de cada una de las habichuelas



%% Pinta train

%%%%%%%%%%%%%%%%%%%%%Se prepara la inicialización%%%%%%%%%%%%%
im_pinta = (imread('train_pinta.jpeg')); % Se cargan los datos para la función, imagen e umbral
um=0.3; % El umbral dependerá de la iluminación de nuestra imagen al igual que la definición de los objetos

[image_pinta,im_reshap_bground_pinta,propied_pinta,A_pinta,contras_pinta,mean_pinta,std_pinta,medias_pinta,mean_medias_pinta,std_medias_pinta]=segmenta(im_pinta,um); % Se llama a la función introduciendo los datos previamente identificados
%De esta función lo mas importante a obtener es contras que es una matriz
%que contará con la varianza de cada una de las habichuelas


%% Test

%%%%%%%%%%%%%%%%%%%%%Se prepara la inicialización%%%%%%%%%%%%%
im_test = (imread('test_habichuelas.jpeg')); % Se cargan los datos para la función, imagen e umbral
um=0.2; % El umbral dependerá de la iluminación de nuestra imagen al igual que la definición de los objetos

[image_test,im_reshap_bground_test,propied_test,A_test,contras_test,mean_test,std_test,medias_test,mean_medias_test,std_medias_test]=segmenta(im_test,um); % Se llama a la función introduciendo los datos previamente identificados
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
close all
figure()
subplot(2,1,1)
histfit(contras_jacome,10,'Normal')
title("Histograma de la varianza de las Jacomelo")
xlabel('Valores de varianza de los pixeles')
ylabel('Cantidad de muestras')

subplot(2,1,2)
histfit(contras_pinta,10,'Normal')
title("Histograma de la varianza de las Pinta")
xlabel('Valores de varianza de los pixeles')
ylabel('Cantidad de muestras')

figure()
subplot(2,1,1)
histfit(medias_jacome,8,'Normal')
title("Histograma de la media de los pixeles de las Jacomelo")
xlabel('Valores de la media de los pixeles')
ylabel('Cantidad de muestras')

subplot(2,1,2)
histfit(medias_pinta,8,'Normal')
title("Histograma de la media de los pixeles de las Pinta")
xlabel('Valores de la media de los pixeles')
ylabel('Cantidad de muestras')

h = [contras_jacome;contras_pinta]';

figure()
hist(h,50)
title("Histograma de la varianzas de las habichuelas")
legend('Jacomelo','Pinta')
xlabel('Valores de varianza de los pixeles')
ylabel('Cantidad de muestras')

m = [medias_jacome;medias_pinta]';

figure()
hist(m,50)
title("Histograma de las medias de las habichuelas")
legend('Jacomelo','Pinta')
xlabel('Valores media de los pixeles')
ylabel('Cantidad de muestras')


% A pesar de que alguno de los histogramas presentan comportamientos que
% pueden aproximarse mejor como una distribucion lognormal, se decidió
% asumir como una distribución normal ya que aparentan tambien tener un
% buen comportamiento asumiendose esta distribucion y mas adelante se
% observan resultados muy buenos utilizando esta distribución

%% Covarianza

covarianza_jaco=cov(contras_jacome,medias_jacome);

covarianza_pinta=cov(contras_pinta,medias_pinta);


% Determinando la matriz de covarianza se puede obtener bastante
% información y es que de la matriz resultante será 2x2 para cada caso,
% tanto para la pinta como la jacomelo, cabe destacar que los feature son
% la varianza de los pixeles de las habichuelas y el area de cada
% habichuela, utilizando Cov(X,Y) se obtiene en la posicion (1,1) = (X,X)
% es decir la varianza del primer feature y en la posicion (2,2) = (Y,Y) se
% obtiene la varianza del segundo feature, las posiciones (1,2) y (2,1) =
% (X,Y)y (Y,X) y por propiedade de la covarianza (X,Y)= (Y,X)

%% algoritmo probando train jacomelo 
label_real= ones(1,100);

figure, imshow(im_jaco)
i=1;

result = ones(1,100);
prob_j = ones(1,100);
prop_p = ones(1,100);
sumatoria = ones(1,100);
prob_j_prueba = ones(1,100);
prob_p_prueba = ones(1,100);

mean_features_jacome = [mean_jacome; mean_medias_jacome]';
mean_features_pinta = [mean_pinta; mean_medias_pinta]';

x_jacome = [contras_jacome;medias_jacome]';


for n=1:size(propied_jaco,1) %En este for del tamaño como cantidad de objetos a los que la funcion regionprops le saca las
%propiedades, se utiliza para poder detectar los objetos en la imagen a
%partir de ahi se asi la estimacion de su es jacomelo o pinta, se encierra
%en un rectangulo si es jacomelo y en otro si es pinta


    if propied_jaco(n).Area>100
            % en esta parte se determinan las probabilidades de que sea que
            % clase
            exponente1 = exp(-(1/2)*((x_jacome(i,:)- mean_features_jacome))*(covarianza_jaco^-1)*(x_jacome(i,:)- mean_features_jacome)');
            prob_j_prueba(i) = (1/(sqrt(2*pi*det(covarianza_jaco))))*exponente1;

            exponente2 = exp(-(1/2)*((x_jacome(i,:)- mean_features_pinta))*(covarianza_pinta^-1)*(x_jacome(i,:)- mean_features_pinta)');
            prob_p_prueba(i) =  (1/(sqrt(2*pi*det(covarianza_pinta))))*exponente2;


            sumatoria = prob_j_prueba(i)+prob_p_prueba(i);
            prob_j(i) = prob_j_prueba(i)/sumatoria;
            prob_p(i) = prob_p_prueba(i)/sumatoria;

            % A partir de aquí esta el algoritmo discriminatoria para
            % seleccionar la clase a la que pertenece
            if prob_j(i)> prob_p(i)
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
% tasa de aciertos utilizando dos feature para el training set de las
% habichuelas jacomelo
aciertos_dosfeat_jaco = sum((result == label_real))/length(contras_jacome)

%% algoritmo probando train Pinta

label_real= ones(1,100);

figure, imshow(im_pinta)
i=1;

result = ones(1,100);
prob_j = ones(1,100);
prop_p = ones(1,100);
prob_p = ones(1,100);
sumatoria = ones(1,100);
prob_j_prueba = ones(1,100);
prob_p_prueba = ones(1,100);

mean_features_jacome = [mean_jacome; mean_medias_jacome]';
mean_features_pinta = [mean_pinta; mean_medias_pinta]';


x_pinta = [contras_pinta;medias_pinta]';

for n=1:size(propied_pinta,1) %En este for del tamaño como cantidad de objetos a los que la funcion regionprops le saca las
%propiedades, se utiliza para poder detectar los objetos en la imagen a
%partir de ahi se asi la estimacion de su es jacomelo o pinta, se encierra
%en un rectangulo si es jacomelo y en otro si es pinta



    if propied_pinta(n).Area>100

            % Se determinan las probabilidades para ambos featues
            exponente1 = exp(-(1/2)*((x_pinta(i,:)- mean_features_jacome))*(covarianza_jaco^-1)*(x_pinta(i,:)- mean_features_jacome)');
            prob_j_prueba(i) =  (1/(sqrt(2*pi*det(covarianza_jaco))))*exponente1  ;

            exponente2 = exp(-(1/2)*((x_pinta(i,:)- mean_features_pinta))*(covarianza_pinta^-1)*(x_pinta(i,:)- mean_features_pinta)');
            prob_p_prueba(i) =  (1/(sqrt(2*pi*det(covarianza_pinta))))*exponente2  ;

            sumatoria = prob_j_prueba(i)+prob_p_prueba(i);
            prob_j(i) = prob_j_prueba(i)/sumatoria;
            prob_p(i) = prob_p_prueba(i)/sumatoria ;
            %------------------------------------------------------------

            if prob_j(i)> prob_p(i)
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

% tasa de aciertos utilizando dos feature para el training set de las
% habichuelas pinta
tasa_aciertos_pinta_dos = sum((result == label_real))/length(contras_jacome)

%% Test clasificación
figure, imshow(im_test)
i=1;

prob_j = ones(1,100);
prop_p = ones(1,100);
prob_p = ones(1,100);
sumatoria = ones(1,100);
prob_j_prueba = ones(1,100);
prob_p_prueba = ones(1,100);

mean_features_jacome = [mean_jacome; mean_medias_jacome]';
mean_features_pinta = [mean_pinta; mean_medias_pinta]';

x_test = [contras_test;medias_test]';


result = ones(1,100);

label_real= [0 1 0 1 1 0 1 1 0 1 1 0 1 1 0 0 0 1 0 0 1 1 1 0 1 0 0 0 1 0  1 0 0 0 0 0 1 0 0 0 1 1 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0 1 1 0 1 1 0 0 0 1 0 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 0 1 0 1 0 1 0 1 1 1];

for n=1:size(propied_test,1) %En este for del tamaño como cantidad de objetos a los que la funcion regionprops le saca las
%propiedades, se utiliza para poder detectar los objetos en la imagen a
%partir de ahi se asi la estimacion de su es jacomelo o pinta, se encierra
%en un rectangulo si es jacomelo y en otro si es pinta

    if propied_test(n).Area>100

            % Se determinan las probabilidades para ambos featues
            exponente1 = exp(-(1/2)*((x_test(i,:)- mean_features_jacome))*(covarianza_jaco^-1)*(x_test(i,:)- mean_features_jacome)');
            prob_j_prueba(i) =  (1/(sqrt(2*pi*det(covarianza_jaco))))*exponente1  ;

            exponente2 = exp(-(1/2)*((x_test(i,:)- mean_features_pinta))*(covarianza_pinta^-1)*(x_test(i,:)- mean_features_pinta)');
            prob_p_prueba(i) =  (1/(sqrt(2*pi*det(covarianza_pinta))))*exponente2  ;

            sumatoria = prob_j_prueba(i)+prob_p_prueba(i);
            prob_j(i) = prob_j_prueba(i)/sumatoria;
            prob_p(i) = prob_p_prueba(i)/sumatoria ;
            %------------------------------------------------------------

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
% Tasa de aciertos para el test set utilizando los dos features
tasa_aciertos_dosfeature = sum((result == label_real))/length(contras_test)
compara = result == label_real;

%% Distribucion 3D
%Histograma en 3D para ver las distribuciones
figure()
X = [contras_jacome' medias_jacome'];
hist3(X,'Nbins',[15 15], 'CDataMode','auto','FaceColor','interp')
xlabel('Varianza')
ylabel('Media')
title('Distribucion de Jacomelos')

figure()
X = [contras_pinta' medias_pinta'];
hist3(X,'Nbins',[15 15], 'CDataMode','auto','FaceColor','interp')
xlabel('Varianza')
ylabel('Media')
title('Distribucion de las Pinta')
