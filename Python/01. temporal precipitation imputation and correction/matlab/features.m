%����ֵ
SaveFolder=strcat('H:\ʱ��Ԥ��\�ĸ������������\ganhan\02_grid_data\xunlian_features\','slope'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end
disp('������...');

%�߳�
fid_1 = fopen('H:\ʱ��Ԥ��\�ĸ������������\ganhan\dem\DEM.txt');
data1 = cell2mat(textscan(fid_1,'%f','headerlines',6));
data1 = reshape(data1,720,240);   %��ͬ�����dem������ Ҫ��
data1 = data1';
fclose(fid_1); 
% % 
% fid_4 = fopen('H:\ʱ��Ԥ��\�ĸ������������\banshirun\dem\MRDEM.txt');
% data4 = cell2mat(textscan(fid_4,'%f','headerlines',6));
% data4 = reshape(data4,720,232);  %��ͬ�����dem������
% data4 = data4';
% fclose(fid_4);
%�¶� ����
fid_2 = fopen('H:\ʱ��Ԥ��\�������\�¶�����\slope.txt');
data2 = cell2mat(textscan(fid_2,'%f','headerlines',6));
data2 = reshape(data2,720,240); %����Ͳ��û��� �¶Ⱥ�������ѡһ������
data2 = data2';
fclose(fid_2);  
%�����������ĸ����������ļ�
fid_3 = fopen('H:\ʱ��Ԥ��\�������\�¶�����\slope.txt');  
data3 = cell2mat(textscan(fid_3,'%f','headerlines',6));
data3 = reshape(data3,720,240);  %����Ͳ��û���
data3 = data3';
fclose(fid_3);  
  
      
count= 1096; %�������� ѵ��1096���߲���365    
data=zeros(count,1); 
for i=1:1:240   %���������Ҳ�Ǹ���dem�������жϵ�  Ҫ��
   for j=1:1:720
       if data1(i,j)~=-9999 && data2(i,j)~=-9999
          for a=1:1:count
              for b=1:1:1
                 data(a,b)=data3(i,j);
              end
          end
      

          SaveFiles=strcat(num2str(i,'%03d'),num2str(j,'%03d')); %CPC����ļ���·��
          SaveFiles=strcat(SaveFiles,'.txt');

         outfile=strcat(SaveFolder,'\',SaveFiles);

         if exist(outfile,'file')~=0 
            delete(outfile);     
         end
         fid1=fopen(outfile,'w');

         for c=1:1:count
             for d=1:1:1
                 if d==1
                     fprintf(fid1,'%g\r\n',data(c,d));
                 else
                     fprintf(fid1,'%g ',data(c,d));
                 end
             end   
         end
         fclose(fid1); 
       end
     
   end
end

disp('�������')


