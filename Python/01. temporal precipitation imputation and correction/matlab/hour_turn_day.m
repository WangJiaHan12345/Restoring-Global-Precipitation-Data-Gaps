% ��Сʱ����תΪ��������
% ����Сʱ�����ļ���
FolderPath=input('���������ݴ洢�ļ���:','s');  %����tif�ļ�������
index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��

SaveFolder=strcat('H:\�й���������\���������-0.1deg\','day'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end

Files=dir(FolderPath);
FilesCount=length(Files);

disp('������...');

% for k= 3:24:FilesCount
%      
%      data_day = zeros(1200,3600);
%      
%      for i = k:k+23        
%          FilePath=strcat(FolderPath,'\',Files(i).name);  %�ļ�·��\�ļ���
%          Name=Files(i).name;
%          location=strfind(Name,'.');
%      
%          fid = fopen(FilePath,'rb','l');
%          data = cell2mat(textscan(fid,'%f','headerlines',6));       
%          data = reshape(data,3600,1200);
%          data = data'; 
%          fclose(fid); 
%          
%          for m = 1:1:1200
%              for n = 1:1:3600
%                  data_day(m,n) = data_day(m,n) + data(m,n);
%              end
%          end
%          
%      end
%      
%      data_day(data_day<0) = 0;
%      
%      SaveFiles=strcat(Name(1:8),'.txt'); %CPC����ļ���·��
%      outfile=strcat(SaveFolder,'\',SaveFiles);
% 
%      if exist(outfile,'file')~=0 
%         delete(outfile);     
%      end
%        
%      fid1=fopen(outfile,'w');
%      fprintf(fid1,'NCOLS        3600\r\nNROWS        1200\r\nXLLCORNER   70\r\nYLLCORNER    15\r\nCELLSIZE    0.100\r\nNODATA_VALUE   -9999.0000\r\n');
%      
%      for a=1:1:440  
%          for b=1:1:700
%              if b==700
%                  fprintf(fid1,'%g\r\n',data_day(a,b));
%              else
%                  fprintf(fid1,'%g ',data_day(a,b));
%              end
%          end   
%      end
%     
%      fclose(fid1);         
% end


%440 700 
for k= 3:24:FilesCount
     
     data_day = zeros(440,700);
     
     for i = k:k+23        
         FilePath=strcat(FolderPath,'\',Files(i).name);  %�ļ�·��\�ļ���
         Name=Files(i).name;
         location=strfind(Name,'.');
     
         fid = fopen(FilePath,'rb','l');
         data = cell2mat(textscan(fid,'%f','headerlines',6));       
         data = reshape(data,700,440);
         data = data'; 
         fclose(fid); 
         
         for m = 1:1:440
             for n = 1:1:700
                 data_day(m,n) = data_day(m,n) + data(m,n);
             end
         end
         
     end
     
     data_day(data_day<0) = 0;
     
     SaveFiles=strcat(Name(1:8),'.txt'); %CPC����ļ���·��
     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
       
     fid1=fopen(outfile,'w');
     fprintf(fid1,'NCOLS        700\r\nNROWS        440\r\nXLLCORNER   70\r\nYLLCORNER    15\r\nCELLSIZE    0.100\r\nNODATA_VALUE   -999.0000\r\n');
     
     for a=1:1:440  
         for b=1:1:700
             if b==700
                 fprintf(fid1,'%g\r\n',data_day(a,b));
             else
                 fprintf(fid1,'%g ',data_day(a,b));
             end
         end   
     end
    
     fclose(fid1);         
end
disp('�������');
