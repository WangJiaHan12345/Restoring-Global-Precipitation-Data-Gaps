%����������תΪȫ������
SaveFolder=strcat('H:\��ظ�ԭ����\�ռ�+ʱ��\01_data\2015-2016\gsmap_gauge\','1-12'); %����ļ���·��
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

%  15-18��
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
%       data1 = reshape(data1,1,368);
%       data1 = data1'; 
%       fclose(fid1);
%       
%       fid2=fopen(FilePath2,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
%       data2 = reshape(data2,1,368);
%       data2 = data2'; 
%       fclose(fid2);
%       
%       fid3=fopen(FilePath3,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
%       data3 = reshape(data3,1,364);
%       data3 = data3'; 
%       fclose(fid3);
%       
%       fid4=fopen(FilePath4,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
%       data4 = reshape(data4,1,361);
%       data4 = data4'; 
%       fclose(fid4);
%       
%       
%       data =zeros(1461,1); % ÿ�������ж�����  ������3-5 276 �ļ���6-8 276 �＾��9-11  273  ������12-2 271
% 
%       
%        a=0; %3-5
%        b=0; %6-8
%        c=0; %9-11
%        d=0; %12-2
%        
%        for i=1:1:59
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%        end
% 
%        for i=60:1:151
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end
% 
%         for i=152:1:243
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%          end
% 
%         for i=244:1:334
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end
%        
%         for i=335:1:425
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end
% 
%        for i=426:1:517
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end
% 
%         for i=518:1:609
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%          end
% 
%         for i=610:1:700
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end
%         
%        for i=701:1:790
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end
% 
%        for i=791:1:882
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end
% 
%         for i=883:1:974
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%          end
% 
%         for i=975:1:1065
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end   
%         
%        for i=1066:1:1155
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%        end       
%         
%         for i=1156:1:1247
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end   
%        
%         for i=1248:1:1339
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%         end   
%         
%         for i=1340:1:1430
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end   
%         
%         for i=1431:1:1461
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end   
%         
%         SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��
%         outfile=strcat(SaveFolder,'\',SaveFiles);      
%         fid=fopen(outfile,'w');      
% 
%          for i=1:1:1461
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

% 15-16
parfor k=3:FilesCount1
      FilePath1=strcat(FolderPath1,'\',Files1(k).name);  %�ļ�·��\�ļ���
      FilePath2=strcat(FolderPath2,'\',Files2(k).name);  %�ļ�·��\�ļ���
      FilePath3=strcat(FolderPath3,'\',Files3(k).name);  %�ļ�·��\�ļ���
      FilePath4=strcat(FolderPath4,'\',Files4(k).name);  %�ļ�·��\�ļ���
      
      Name=Files1(k).name;
      location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
      

      fid1=fopen(FilePath1,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data1 = cell2mat(textscan(fid1,'%f','headerlines',0));
      data1 = reshape(data1,1,184);
      data1 = data1'; 
      fclose(fid1);
      
      fid2=fopen(FilePath2,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
      data2 = reshape(data2,1,184);
      data2 = data2'; 
      fclose(fid2);
      
      fid3=fopen(FilePath3,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
      data3 = reshape(data3,1,182);
      data3 = data3'; 
      fclose(fid3);
      
      fid4=fopen(FilePath4,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
      data4 = reshape(data4,1,181);
      data4 = data4'; 
      fclose(fid4);
      
      
      data =zeros(731,1); % ÿ�������ж�����  ������3-5 276 �ļ���6-8 276 �＾��9-11  273  ������12-2 271

      
       a=0; %3-5
       b=0; %6-8
       c=0; %9-11
       d=0; %12-2
       
       for i=1:1:59
           for j=1:1:1
               d=d+1;
               data(i,1)=data4(d,j);
           end
       end

       for i=60:1:151
           for j=1:1:1
               a=a+1;
               data(i,1)=data1(a,j);
           end
        end

        for i=152:1:243
           for j=1:1:1
               b=b+1;
               data(i,1)=data2(b,j);
           end
         end

        for i=244:1:334
           for j=1:1:1
               c=c+1;
               data(i,1)=data3(c,j);
           end
        end
       
        for i=335:1:425
           for j=1:1:1
               d=d+1;
               data(i,1)=data4(d,j);
           end
        end

       for i=426:1:517
           for j=1:1:1
               a=a+1;
               data(i,1)=data1(a,j);
           end
        end

        for i=518:1:609
           for j=1:1:1
               b=b+1;
               data(i,1)=data2(b,j);
           end
         end

        for i=610:1:700
           for j=1:1:1
               c=c+1;
               data(i,1)=data3(c,j);
           end
        end
        
       for i=701:1:731
           for j=1:1:1
               d=d+1;
               data(i,1)=data4(d,j);
           end
        end

        
        SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��
        outfile=strcat(SaveFolder,'\',SaveFiles);      
        fid=fopen(outfile,'w');      

         for i=1:1:731
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
%       data1 = reshape(data1,1,90);
%       data1 = data1'; 
%       fclose(fid1);
%       
%       fid2=fopen(FilePath2,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
%       data2 = reshape(data2,1,90);
%       data2 = data2'; 
%       fclose(fid2);
%       
%       fid3=fopen(FilePath3,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
%       data3 = reshape(data3,1,90);
%       data3 = data3'; 
%       fclose(fid3);
%       
%       fid4=fopen(FilePath4,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
%       data4 = reshape(data4,1,85);
%       data4 = data4'; 
%       fclose(fid4);
%       
%       
%       data =zeros(355,1); % ÿ�������ж�����  ������3-5 276 �ļ���6-8 276 �＾��9-11  273  ������12-2 271
% 
%       
%        a=0; %3-5
%        b=0; %6-8
%        c=0; %9-11
%        d=0; %12-2
%        
%        for i=1:1:55
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%        end
% 
%        for i=56:1:145
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%         end
% 
%         for i=146:1:235
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%          end
% 
%         for i=236:1:325
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%         end
%        
%         for i=326:1:355
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end
% 
%         
%         SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��
%         outfile=strcat(SaveFolder,'\',SaveFiles);      
%         fid=fopen(outfile,'w');      
% 
%          for i=1:1:355
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

% 16��
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
%       data1 = reshape(data1,1,30);
%       data1 = data1'; 
%       fclose(fid1);
%       
%       fid2=fopen(FilePath2,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
%       data2 = reshape(data2,1,30);
%       data2 = data2'; 
%       fclose(fid2);
%       
%       fid3=fopen(FilePath3,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
%       data3 = reshape(data3,1,30);
%       data3 = data3'; 
%       fclose(fid3);
%       
%       fid4=fopen(FilePath4,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
%       data4 = reshape(data4,1,30);
%       data4 = data4'; 
%       fclose(fid4);
%       
%       
%       data =zeros(120,1); % ÿ�������ж�����  ������3-5 276 �ļ���6-8 276 �＾��9-11  273  ������12-2 271
% 
%       
%        a=0; %3-5
%        b=0; %6-8
%        c=0; %9-11
%        d=0; %12-2
%        
%        for i=1:1:30
%            for j=1:1:1
%                a=a+1;
%                data(i,1)=data1(a,j);
%            end
%        end
% 
%        for i=31:1:60
%            for j=1:1:1
%                b=b+1;
%                data(i,1)=data2(b,j);
%            end
%         end
% 
%         for i=61:1:90
%            for j=1:1:1
%                c=c+1;
%                data(i,1)=data3(c,j);
%            end
%          end
% 
%         for i=91:1:120
%            for j=1:1:1
%                d=d+1;
%                data(i,1)=data4(d,j);
%            end
%         end
%        
%         
%         SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��
%         outfile=strcat(SaveFolder,'\',SaveFiles);      
%         fid=fopen(outfile,'w');      
% 
%          for i=1:1:120
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
