function [image,im_reshap_bground,propied,acu,contras,mean_train,std_train,medias,mean_medias,std_medias]=segmenta(im,level)

%Función para segmentar la imagen, sacar propiedades de la misma y
%determinar la varianza de los objetos en la imagen

% Segmentacion y selecciona los objetos  

image= rgb2gray(im); %Imagen pasada a escala de grises
%Se indica un nivel de Threshold medio, este es relativo a propiedades de la imagen como la intensidad de la luz
bw= im2bw(im,level); %Binarizar la imagen, en base al threshold seleccionado se
%designa que será negro y que será blanco para convertir la imagen de
%escala de ingres en una imagen binaria
%%%%%%%%% poner fondo totalmente negro
[f, c]= size(image); %Se extrae el tamaño de la imagen en escala de grises
im_reshap= (image(:))'; %Se convierte en un vector
for i=1:length(im_reshap) %En este for se procede a convertir en 0 (negro total) cualquier valor menor a 120
    if im_reshap(1,i)<100
        im_reshap(1,i)=0;
    end
end

im_reshap_bground= reshape(im_reshap,f,c); %Ahora con el fondo totalmente negro se le devuelve su forma original

figure, imshow(im); %Se presenta la imagen en escala de grises con el fondo negro

%Etiqueta de elementos
[L, Ne]=bwlabel(bw); %función que etiqueta componentes en una imagen binaria 2D
propied=regionprops(L); %Se utilizar regionprops para obtener las propiedades de la imagen etiquetada
i=1;
contras=ones(1,100);
medias=ones(1,100);
area_feature=ones(1,100);

[f, c]= size(propied);

for n=1:size(propied,1) %En este for del tamaño como cantidad de objetos a los que la funcion regionprops les
    %saco las propiedades, se realiza un if donde se tomarán en cuanta los
    %objetos con areas mayores a 100, aqui dentro se utiliza la funcion
    %rectangle para encerrar estos objetos, con la variable acu se aisla
    %cada objeto de la imagen, los pixeles con valor de 0 (el fondo
    %ngegro) ahora se eliminan para tener solo información de las
    %habichuelas y en la variable contras se van acumulando las varianzas
    %de cada habichuela, como segundo feature se tomo el area de cada
    %habichuela a apoyandonos de la funcion regionprops que nos da esta
    %propiedad y la funcion imcrop para poder evaluar cada habichuela por
    %separado
    if propied(n).Area>100
    rectangle('Position',propied(n).BoundingBox,'EdgeColor','g','LineWidth',2);
    acu= double(imcrop(im_reshap_bground,propied(n).BoundingBox));
    acu(acu==0)=[];
    contras(i)= var(acu);
    medias(i)= mean(acu);
    i=i+1;
    end
    
end

%%%%%%%%% Se consigue la media y la varianza del feature%%%%%%%%%%%%%%

mean_train = mean(contras);
std_train = std(contras);

%%%%%%%%% Se consigue la media y la varianza del segundo feature que es la media de los pixeles%%%%%%%%%%%%%%

mean_medias = mean(medias);
std_medias = std(medias);


