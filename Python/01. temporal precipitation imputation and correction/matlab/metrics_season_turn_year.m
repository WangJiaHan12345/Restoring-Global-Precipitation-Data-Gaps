
SaveFolder=strcat('G:\��ҵ����ͼ\��ظ�ԭ\�ռ�\�·���������ͼ\','1-12'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end

FolderPath1=input('���������ݴ洢�ļ���:','s'); %����3-5����
index1=strfind(FolderPath1,'\');  %����ַ�'\'��FolderPath��λ��
Files1=dir(FolderPath1);
FilesCount1=length(Files1);

FolderPath2=input('���������ݴ洢�ļ���:','s'); %����6-8����
index2=strfind(FolderPath2,'\');  %����ַ�'\'��FolderPath��λ��
Files2=dir(FolderPath2);
FilesCount2=length(Files2);

FolderPath3=input('���������ݴ洢�ļ���:','s'); %����9-11����
index3=strfind(FolderPath3,'\');  %����ַ�'\'��FolderPath��λ��
Files3=dir(FolderPath3);
FilesCount3=length(Files3);

FolderPath4=input('���������ݴ洢�ļ���:','s'); %����12-2����
index4=strfind(FolderPath4,'\');  %����ַ�'\'��FolderPath��λ��
Files4=dir(FolderPath4);
FilesCount4=length(Files4);

disp('������...');

 %15-18��
% parfor k=3:FilesCount1
%       FilePath1=strcat(FolderPath1,'\',Files1(k).name);  %�ļ�·��\�ļ���
%       FilePath2=strcat(FolderPath2,'\',Files2(k).name);  %�ļ�·��\�ļ���
%       FilePath3=strcat(FolderPath3,'\',Files3(k).name);  %�ļ�·��\�ļ���
%       FilePath4=strcat(FolderPath4,'\',Files4(k).name);  %�ļ�·��\�ļ���
%       
%       Name=Files1(k).name;
%       location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
%       
% 
%       fid1=fopen(FilePath1,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data1 = cell2mat(textscan(fid1,'%f','headerlines',0));
%       data1 = reshape(data1,1,12);
%       data1 = data1'; 
%       fclose(fid1);
%       
%       fid2=fopen(FilePath2,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
%       data2 = reshape(data2,1,12);
%       data2 = data2'; 
%       fclose(fid2);
%       
%       fid3=fopen(FilePath3,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
%       data3 = reshape(data3,1,12);
%       data3 = data3'; 
%       fclose(fid3);
%       
%       fid4=fopen(FilePath4,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
%       data4 = reshape(data4,1,12);
%       data4 = data4'; 
%       fclose(fid4);
%       
%       
%       data =zeros(48,1); % ÿ�������ж�����  ������3-5 276 �ļ���6-8 276 �＾��9-11  273  ������12-2 271
% 
%       
%        a=0; %3-5
%        b=0; %6-8
%        c=0; %9-11
%        d=0; %12-2
%        
%        for i=1:1:2
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%        end
% 
%        for i=3:1:5
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end
% 
%         for i=6:1:8
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%          end
% 
%         for i=9:1:11
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end
%        
%         for i=12:1:14
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end
% 
%        for i=15:1:17
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end
% 
%         for i=18:1:20
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%          end
% 
%         for i=21:1:23
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end
%         
%        for i=24:1:26
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end
% 
%        for i=27:1:29
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end
% 
%         for i=30:1:32
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%          end
% 
%         for i=33:1:35
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end   
%         
%        for i=36:1:38
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%        end       
%         
%         for i=39:1:41
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end   
%        
%         for i=42:1:44
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%         end   
%         
%         for i=45:1:47
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end   
%         
%         for i=48:1:48
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end   
%         
%         SaveFiles=strcat(Name(1:location(end)-1),'.txt'); %CPC����ļ���·��
%         outfile=strcat(SaveFolder,'\',SaveFiles);      
%         fid=fopen(outfile,'w');      
% 
%          for i=1:1:48
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid,'%g\r\n',data(i,j));
%                  else
%                     fprintf(fid,'%g ',data(i,j));
%                  end
%              end   
%          end
%          fclose(fid); 
%                
% end


%18
% parfor k=3:FilesCount1
%       FilePath1=strcat(FolderPath1,'\',Files1(k).name);  %�ļ�·��\�ļ���
%       FilePath2=strcat(FolderPath2,'\',Files2(k).name);  %�ļ�·��\�ļ���
%       FilePath3=strcat(FolderPath3,'\',Files3(k).name);  %�ļ�·��\�ļ���
%       FilePath4=strcat(FolderPath4,'\',Files4(k).name);  %�ļ�·��\�ļ���
%       
%       Name=Files1(k).name;
%       location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
%       
% 
%       fid1=fopen(FilePath1,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data1 = cell2mat(textscan(fid1,'%f','headerlines',0));
%       data1 = reshape(data1,1,3);
%       data1 = data1'; 
%       fclose(fid1);
%       
%       fid2=fopen(FilePath2,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
%       data2 = reshape(data2,1,3);
%       data2 = data2'; 
%       fclose(fid2);
%       
%       fid3=fopen(FilePath3,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
%       data3 = reshape(data3,1,3);
%       data3 = data3'; 
%       fclose(fid3);
%       
%       fid4=fopen(FilePath4,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
%       data4 = reshape(data4,1,3);
%       data4 = data4'; 
%       fclose(fid4);
%       
%       
%       data =zeros(12,1); % ÿ�������ж�����  ������3-5 276 �ļ���6-8 276 �＾��9-11  273  ������12-2 271
% 
%       
%        a=0; %3-5
%        b=0; %6-8
%        c=0; %9-11
%        d=0; %12-2
%        
%        for i=1:1:2
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%        end
% 
%        for i=3:1:5
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end
% 
%         for i=6:1:8
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%          end
% 
%         for i=9:1:11
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end
%        
%         for i=12:1:12
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end
%         
%         SaveFiles=strcat(Name(1:location(end)-1),'.txt'); %CPC����ļ���·��
%         outfile=strcat(SaveFolder,'\',SaveFiles);      
%         fid=fopen(outfile,'w');      
% 
%          for i=1:1:12
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid,'%g\r\n',data(i,j));
%                  else
%                     fprintf(fid,'%g ',data(i,j));
%                  end
%              end   
%          end
%          fclose(fid); 
%                
% end
% disp('�������')


%15-16
parfor k=3:FilesCount1
      FilePath1=strcat(FolderPath1,'\',Files1(k).name);  %�ļ�·��\�ļ���
      FilePath2=strcat(FolderPath2,'\',Files2(k).name);  %�ļ�·��\�ļ���
      FilePath3=strcat(FolderPath3,'\',Files3(k).name);  %�ļ�·��\�ļ���
      FilePath4=strcat(FolderPath4,'\',Files4(k).name);  %�ļ�·��\�ļ���
      
      Name=Files1(k).name;
      location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
      

      fid1=fopen(FilePath1,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data1 = cell2mat(textscan(fid1,'%f','headerlines',0));
      data1 = reshape(data1,1,6);
      data1 = data1'; 
      fclose(fid1);
      
      fid2=fopen(FilePath2,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
      data2 = reshape(data2,1,6);
      data2 = data2'; 
      fclose(fid2);
      
      fid3=fopen(FilePath3,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
      data3 = reshape(data3,1,6);
      data3 = data3'; 
      fclose(fid3);
      
      fid4=fopen(FilePath4,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
      data4 = reshape(data4,1,6);
      data4 = data4'; 
      fclose(fid4);
      
      
      data =zeros(24,1); 

      
       a=0; %3-5
       b=0; %6-8
       c=0; %9-11
       d=0; %12-2
       
       for i=1:1:2
           for j=1:1:1
               d=d+1;
               data(i,1)=data4(d,j);
           end
       end

       for i=3:1:5
           for j=1:1:1
               a=a+1;
               data(i,1)=data1(a,j);
           end
        end

        for i=6:1:8
           for j=1:1:1
               b=b+1;
               data(i,1)=data2(b,j);
           end
         end

        for i=9:1:11
           for j=1:1:1
               c=c+1;
               data(i,1)=data3(c,j);
           end
        end
       
        for i=12:1:14
           for j=1:1:1
               d=d+1;
               data(i,1)=data4(d,j);
           end
        end
        
         for i=15:1:17
           for j=1:1:1
               a=a+1;
               data(i,1)=data1(a,j);
           end
        end

        for i=18:1:20
           for j=1:1:1
               b=b+1;
               data(i,1)=data2(b,j);
           end
         end

        for i=21:1:23
           for j=1:1:1
               c=c+1;
               data(i,1)=data3(c,j);
           end
        end
       
        for i=24:1:24
           for j=1:1:1
               d=d+1;
               data(i,1)=data4(d,j);
           end
        end
        
        SaveFiles=strcat(Name(1:location(end)-1),'.txt'); %CPC����ļ���·��
        outfile=strcat(SaveFolder,'\',SaveFiles);      
        fid=fopen(outfile,'w');      

         for i=1:1:24
             for j=1:1:1
                 if j==1
                     fprintf(fid,'%g\r\n',data(i,j));
                 else
                    fprintf(fid,'%g ',data(i,j));
                 end
             end   
         end
         fclose(fid); 
               
end
disp('�������')