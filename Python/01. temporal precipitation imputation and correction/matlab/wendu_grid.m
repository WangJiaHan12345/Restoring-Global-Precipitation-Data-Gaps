%���¶�����ת��Ϊ��������
%���� �¶������ļ�
FolderPath1=input('���������ݴ洢�ļ���:','s'); 
index1=strfind(FolderPath1,'\');  %����ַ�'\'��FolderPath��λ��
SaveFolder=strcat('H:\ʱ��Ԥ��\�ĸ������������\ganhan\02_grid_data\ceshi_features\','wendu'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end
Files1=dir(FolderPath1);
FilesCount1=length(Files1);

% ������Ա�ʾ������������ݣ����������dem
% H:\ʱ��Ԥ��\�ĸ������������\shirun\02_grid_data\xunlian_features\dem
FolderPath2=input('���������ݴ洢�ļ���:','s'); 
index2=strfind(FolderPath2,'\');  %����ַ�'\'��FolderPath��λ��                                  
Files2=dir(FolderPath2);
FilesCount2=length(Files2);

disp('������...');


for k=3:FilesCount2
      Name=Files2(k).name;
      location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
      
      
      i= str2num(Name(location(end)-6:location(end)-4));
      j= str2num(Name(location(end)-3:location(end)-1));
      
      
      result = zeros(365,1);  % 365  1096
      
      parfor m =3:FilesCount1
          FilePath = strcat(FolderPath1,'\',Files1(m).name);
          fid = fopen(FilePath,'rb','l');
          data = cell2mat(textscan(fid,'%f','headerlines',6));
          data = reshape(data,720,240);
          data = data'; 
          
          result(m-2,1)=data(i,j); 
          
          fclose(fid); 
          
      end
      
     SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��

     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     
     
     fid1=fopen(outfile,'w');
     for i=1:1:365  % 365
         for j=1:1:1
             if j==1
                 fprintf(fid1,'%g\r\n',result(i,j));
             else
                fprintf(fid1,'%g ',result(i,j));
             end
         end   
     end
     fclose(fid1);         
end

disp('�������');