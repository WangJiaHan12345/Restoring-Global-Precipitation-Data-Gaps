%����������ݷּ���
FolderPath=input('���������ݴ洢�ļ���:','s'); %����
index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��
SaveFolder=strcat('H:\ʱ��Ԥ��\������\shirun\02_final_data\����\��\6-8\','xunlian\Early'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('������...');

 

%xunlain
for k=3:FilesCount
      FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
      Name=Files(k).name;
      location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
  
      fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data = cell2mat(textscan(fid,'%f','headerlines',0));
      data = reshape(data,1,1096);
      data = data';
      
      data1=zeros(276,1); %%%%%%%%%%%%%%%%%%%%
      
       a=0;
       for i=152:1:243
           for j=1:1:1
               a=a+1;
               data1(a,1)=data(i,j);
           end
       end
        
        for i=518:1:609
           for j=1:1:1
               a=a+1;
               data1(a,1)=data(i,j);
           end
        end
        
         for i=883:1:974
           for j=1:1:1
               a=a+1;
               data1(a,1)=data(i,j);
           end
         end
         
     SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��
     %SaveFiles=strcat(Name(1:location(end)-1),'.txt');
    
     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     fid1=fopen(outfile,'w');
    
     
     for i=1:1:276
         for j=1:1:1
             if j==1
                 fprintf(fid1,'%g\r\n',data1(i,j));
             else
                fprintf(fid1,'%g ',data1(i,j));
             end
         end   
     end
     fclose(fid1); 
     fclose(fid); 
        
         
end

% %ceshi
% for k=3:FilesCount
%       FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
%       Name=Files(k).name;
%       location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
%   
%       fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data = cell2mat(textscan(fid,'%f','headerlines',0));
%       data = reshape(data,1,365);
%       data = data';
%       
%       data1=zeros(92,1); %%%%%%%%%%%%%%%%%%%%
%       
%        a=0;
%        for i=60:1:151
%            for j=1:1:1
%                a=a+1;
%                data1(a,1)=data(i,j);
%            end
%        end
%        
%      SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��
%      %SaveFiles=strcat(Name(1:location(end)-1),'.txt');
%     
%      outfile=strcat(SaveFolder,'\',SaveFiles);
% 
%      if exist(outfile,'file')~=0 
%         delete(outfile);     
%      end
%      fid1=fopen(outfile,'w');
%     
%      
%      for i=1:1:92
%          for j=1:1:1
%              if j==1
%                  fprintf(fid1,'%g\r\n',data1(i,j));
%              else
%                 fprintf(fid1,'%g ',data1(i,j));
%              end
%          end   
%      end
%      fclose(fid1); 
%      fclose(fid); 
%         
%          
% end
 disp('�������')


 