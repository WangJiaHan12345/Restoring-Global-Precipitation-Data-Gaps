% %���Խ��תΪÿ�ո�������
% SaveFolder=strcat('H:\ʱ��Ԥ��\������\24��\result\','predict'); %����ļ���·��
% if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
%     mkdir(SaveFolder);
% end
% 
% disp('������...');
% 
% fid_2 = fopen('H:\ʱ��Ԥ��\������\24��\result\predict\out_data.txt');
% data1 = cell2mat(textscan(fid_2,'%f','headerlines',0));
% data1 = reshape(data1,1,85775);
% data1 = data1';
% fclose(fid_2);  
% 
% grid_count=235;
% a=0;
% for k=1:1:365 %2018�������
%    data2=zeros(grid_count,1); 
%   
%     for i=1:1:grid_count
%        for j=1:1:1
%             a=a+1;
%             data2(i,j)=data1(a,j); 
%        end
%     end
% 
%     SaveFiles=strcat(num2str(k),'.txt'); %CPC����ļ���·��
% 
% 
%     outfile=strcat(SaveFolder,'\',SaveFiles);
% 
%     if exist(outfile,'file')~=0 
%     delete(outfile);     
%     end
%     fid1=fopen(outfile,'w');
% 
% 
%     for i=1:1:grid_count
%      for j=1:1:1
%          if j==1
%              fprintf(fid1,'%g\r\n',data2(i,j));
%          else
%             fprintf(fid1,'%g ',data2(i,j));
%          end
%      end   
%     end
%     fclose(fid1); 
%  
% 
% end
% disp('�������')
% 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%dayתΪgrid
FolderPath=input('���������ݴ洢�ļ���:','s'); %day
index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��
SaveFolder=strcat('H:\ʱ��Ԥ��\������\23��\02_grid_data\','Final\ceshi'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('������...');

day_count=365;  %1096  365
grid_count=562;

fid_2 = fopen('H:\ʱ��Ԥ��\������\23��\dem\DEM.txt');  %�ҵ�����λ��
data1 = cell2mat(textscan(fid_2,'%f','headerlines',6));
data1 = reshape(data1,720,229);
data1 = data1';
fclose(fid_2);  

grid=zeros(grid_count,2);

a=0;
for i=1:1:229
    for j=1:1:720
        if data1(i,j)~=-9999
            a=a+1;
            grid(a,1)=i;
            grid(a,2)=j;
        end
    end
end

            
           
for i=1:1:grid_count
    data2=zeros(day_count,1); 
    a=0;
    for k=3:FilesCount
          a=a+1;
          FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
          Name=Files(k).name;
          location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��

          fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
          data = cell2mat(textscan(fid,'%f','headerlines',0));
          data = reshape(data,1,grid_count);
          data = data';
          fclose(fid); 
          
          data2(a,1)=data(i,1);     
    end
    
    
    SaveFiles=strcat(num2str(grid(i,1),'%03d'),num2str(grid(i,2),'%03d')); %CPC����ļ���·��
    SaveFiles=strcat(SaveFiles,'.txt'); %CPC����ļ���·��
    outfile=strcat(SaveFolder,'\',SaveFiles);

    if exist(outfile,'file')~=0 
    delete(outfile);     
    end
    fid1=fopen(outfile,'w');


     for i=1:1:day_count
         for j=1:1:1
             if j==1
                 fprintf(fid1,'%g\r\n',data2(i,j));
             else
                 fprintf(fid1,'%g ',data2(i,j));
             end
         end 
     end
     fclose(fid1); 

end
  
disp('�������')


 
