fid=fopen('H:\�ռ�Ԥ��\shirun\dem\global\DEM.txt','rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
data = cell2mat(textscan(fid,'%f','headerlines',6));   %ȥ��ǰ6��
data = reshape(data,720,229); % ��ת
data = data';

SaveFolder=strcat('H:\ʱ��Ԥ��\24��\','dem'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end

disp('������...');

fid_2 = fopen('H:\ʱ��Ԥ��\28����\24.txt');
data1 = cell2mat(textscan(fid_2,'%f','headerlines',6));
data1 = reshape(data1,720,240);
data1 = data1';
fclose(fid_2);  

data2=zeros(229,720); %%%%%%%%%%%%%%%%%%%%%%%


for i=1:1:229
   for j=1:1:720
       if data1(i,j)~=-999
          data2(i,j)=data(i,j);
       else
           data2(i,j)=-9999;
       end
   end
end



SaveFiles=strcat('DEM','.txt'); %CPC����ļ���·��
 %SaveFiles=strcat(Name(1:location(end)-1),'.txt');

 outfile=strcat(SaveFolder,'\',SaveFiles);


if exist(outfile,'file')~=0 
delete(outfile);     
end
fid1=fopen(outfile,'w');

fprintf(fid1,'NCOLS        720\r\nNROWS        229\r\nXLLCORNER   0\r\nYLLCORNER    -60\r\nCELLSIZE    0.5\r\nNODATA_VALUE    -9999\r\n');


for i=1:1:229
 for j=1:1:720
     if j==720
         fprintf(fid1,'%g\r\n',data2(i,j));
     else
        fprintf(fid1,'%g ',data2(i,j));
     end
 end   
end
fclose(fid1); 
fclose(fid); 

disp('�������')


