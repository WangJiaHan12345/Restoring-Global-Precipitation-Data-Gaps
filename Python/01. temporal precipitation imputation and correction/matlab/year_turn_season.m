%��ȫ������תΪ��������
FolderPath=input('���������ݴ洢�ļ���:','s'); %����ȫ������
index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��
SaveFolder1=strcat('H:\��ظ�ԭ����\ʱ��Ԥ��\2015-2016\01_clip_data\����\','gsmap_rnt\12-2'); %����ļ���·��
if exist(SaveFolder1,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder1);
end

SaveFolder2=strcat('H:\��ظ�ԭ����\ʱ��Ԥ��\2015-2016\01_clip_data\����\','gsmap_rnt\3-5'); %����ļ���·��
if exist(SaveFolder2,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder2);
end

SaveFolder3=strcat('H:\��ظ�ԭ����\ʱ��Ԥ��\2015-2016\01_clip_data\����\','gsmap_rnt\6-8'); %����ļ���·��
if exist(SaveFolder3,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder3);
end

SaveFolder4=strcat('H:\��ظ�ԭ����\ʱ��Ԥ��\2015-2016\01_clip_data\����\','gsmap_rnt\9-11'); %����ļ���·��
if exist(SaveFolder4,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder4);
end

Files=dir(FolderPath);
FilesCount=length(Files);
disp('������...');

% xunlain  15-17��
% parfor k=3:FilesCount
%       FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
%       Name=Files(k).name;
%       location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
%   
%       i= str2num(Name(location(end)-6:location(end)-4));
%       j= str2num(Name(location(end)-3:location(end)-1));
% 
%       fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data = cell2mat(textscan(fid,'%f','headerlines',0));
%       data = reshape(data,1,1096);
%       data = data'; 
%       
%       
%       data_1 =zeros(271,1); % ÿ�������ж�����  ������3-5 276 �ļ���6-8 276 �＾��9-11  273  ������12-2 271
%       data_2 =zeros(276,1);
%       data_3 =zeros(276,1);
%       data_4 =zeros(273,1);
%       
%        a=0;
%        b=0;
%        c=0;
%        d=0;
%        
%        for i=1:1:59
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%        end
% 
%        for i=60:1:151
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=152:1:243
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=244:1:334
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end
%        
%         for i=335:1:425
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end
% 
%        for i=426:1:517
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=518:1:609
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=610:1:700
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end
%         
%        for i=701:1:790
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end
% 
%        for i=791:1:882
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=883:1:974
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=975:1:1065
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end   
%         
%        for i=1066:1:1096
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end        
%         
% 
%         SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��
% 
%         outfile1=strcat(SaveFolder1,'\',SaveFiles);
%         outfile2=strcat(SaveFolder2,'\',SaveFiles);
%         outfile3=strcat(SaveFolder3,'\',SaveFiles);
%         outfile4=strcat(SaveFolder4,'\',SaveFiles);
% 
%         
%         fid1=fopen(outfile1,'w');
%         fid2=fopen(outfile2,'w');
%         fid3=fopen(outfile3,'w');
%         fid4=fopen(outfile4,'w');
%         
% 
%          for i=1:1:271
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid1,'%g\r\n',data_1(i,j));
%                  else
%                     fprintf(fid1,'%g ',data_1(i,j));
%                  end
%              end   
%          end
%          fclose(fid1); 
%          
%          for i=1:1:276
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid2,'%g\r\n',data_2(i,j));
%                  else
%                     fprintf(fid2,'%g ',data_2(i,j));
%                  end
%              end   
%          end   
%          fclose(fid2);
%          
%          for i=1:1:276
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid3,'%g\r\n',data_3(i,j));
%                  else
%                     fprintf(fid3,'%g ',data_3(i,j));
%                  end
%              end   
%          end      
%          fclose(fid3); 
%          
%          for i=1:1:273
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid4,'%g\r\n',data_4(i,j));
%                  else
%                     fprintf(fid4,'%g ',data_4(i,j));
%                  end
%              end   
%          end         
%          
%          fclose(fid4); 
%          fclose(fid); 
%                
% end
% disp('�������')


%ceshi  18��
% parfor k=3:FilesCount
%       FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
%       Name=Files(k).name;
%       location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
%   
%       i= str2num(Name(location(end)-6:location(end)-4));
%       j= str2num(Name(location(end)-3:location(end)-1));
% 
%       fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data = cell2mat(textscan(fid,'%f','headerlines',0));
%       data = reshape(data,1,365);
%       data = data'; 
%       
%       
%       data_1 =zeros(90,1); % ÿ�������ж�����  ������3-5 276 �ļ���6-8 276 �＾��9-11  273  ������12-2 271
%       data_2 =zeros(92,1);
%       data_3 =zeros(92,1);
%       data_4 =zeros(91,1);
%       
%        a=0;
%        b=0;
%        c=0;
%        d=0;
%        
%        for i=1:1:59
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%        end
% 
%        for i=60:1:151
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=152:1:243
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=244:1:334
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end
%        
%         for i=335:1:365
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end
%         
% 
%         SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��
% 
%         outfile1=strcat(SaveFolder1,'\',SaveFiles);
%         outfile2=strcat(SaveFolder2,'\',SaveFiles);
%         outfile3=strcat(SaveFolder3,'\',SaveFiles);
%         outfile4=strcat(SaveFolder4,'\',SaveFiles);
% 
%         
%         fid1=fopen(outfile1,'w');
%         fid2=fopen(outfile2,'w');
%         fid3=fopen(outfile3,'w');
%         fid4=fopen(outfile4,'w');
%         
% 
%          for i=1:1:90
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid1,'%g\r\n',data_1(i,j));
%                  else
%                     fprintf(fid1,'%g ',data_1(i,j));
%                  end
%              end   
%          end
%          
%          for i=1:1:92
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid2,'%g\r\n',data_2(i,j));
%                  else
%                     fprintf(fid2,'%g ',data_2(i,j));
%                  end
%              end   
%          end         
%          
%          for i=1:1:92
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid3,'%g\r\n',data_3(i,j));
%                  else
%                     fprintf(fid3,'%g ',data_3(i,j));
%                  end
%              end   
%          end         
%          
%          for i=1:1:91
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid4,'%g\r\n',data_4(i,j));
%                  else
%                     fprintf(fid4,'%g ',data_4(i,j));
%                  end
%              end   
%          end            
%          
%          fclose(fid1); 
%          fclose(fid2); 
%          fclose(fid3); 
%          fclose(fid4); 
%          fclose(fid); 
%                
% end
% disp('�������')


% 15-18��
% for k=3:FilesCount
%       FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
%       Name=Files(k).name;
%       location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
%   
%       i= str2num(Name(location(end)-6:location(end)-4));
%       j= str2num(Name(location(end)-3:location(end)-1));
% 
%       fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data = cell2mat(textscan(fid,'%f','headerlines',0));
%       data = reshape(data,1,1461);
%       data = data'; 
%       
%       
%       data_1 =zeros(361,1); % ÿ�������ж�����  ������3-5 276 �ļ���6-8 276 �＾��9-11  273  ������12-2 271
%       data_2 =zeros(368,1);
%       data_3 =zeros(368,1);
%       data_4 =zeros(364,1);
%       
%        a=0;
%        b=0;
%        c=0;
%        d=0;
%        
%        for i=1:1:59
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%        end
% 
%        for i=60:1:151
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=152:1:243
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=244:1:334
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end
%        
%         for i=335:1:425
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end
% 
%        for i=426:1:517
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=518:1:609
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=610:1:700
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end
%         
%        for i=701:1:790
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end
% 
%        for i=791:1:882
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=883:1:974
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=975:1:1065
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end   
%         
%        for i=1066:1:1155
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%        end       
%         
%        for i=1156:1:1247
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%        end  
%         
%        for i=1248:1:1339
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%        end  
%         
%                 
%        for i=1340:1:1430
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%        end  
%         
%        for i=1431:1:1461
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end  
%         
% 
%         SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��
% 
%         outfile1=strcat(SaveFolder1,'\',SaveFiles);
%         outfile2=strcat(SaveFolder2,'\',SaveFiles);
%         outfile3=strcat(SaveFolder3,'\',SaveFiles);
%         outfile4=strcat(SaveFolder4,'\',SaveFiles);
% 
%         
%         fid1=fopen(outfile1,'w');
%         fid2=fopen(outfile2,'w');
%         fid3=fopen(outfile3,'w');
%         fid4=fopen(outfile4,'w');
%         
% 
%          for i=1:1:361
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid1,'%g\r\n',data_1(i,j));
%                  else
%                     fprintf(fid1,'%g ',data_1(i,j));
%                  end
%              end   
%          end
%          fclose(fid1); 
%          
%          for i=1:1:368
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid2,'%g\r\n',data_2(i,j));
%                  else
%                     fprintf(fid2,'%g ',data_2(i,j));
%                  end
%              end   
%          end   
%          fclose(fid2);
%          
%          for i=1:1:368
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid3,'%g\r\n',data_3(i,j));
%                  else
%                     fprintf(fid3,'%g ',data_3(i,j));
%                  end
%              end   
%          end      
%          fclose(fid3); 
%          
%          for i=1:1:364
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid4,'%g\r\n',data_4(i,j));
%                  else
%                     fprintf(fid4,'%g ',data_4(i,j));
%                  end
%              end   
%          end         
%          
%          fclose(fid4); 
%          fclose(fid); 
%                
% end
% disp('�������')

%15-16��
parfor k=3:FilesCount
      FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
      Name=Files(k).name;
      location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
  
%       i= str2num(Name(location(end)-6:location(end)-4));
%       j= str2num(Name(location(end)-3:location(end)-1));

      fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data = cell2mat(textscan(fid,'%f','headerlines',0));
      data = reshape(data,1,731);
      data = data'; 
      fclose(fid);
      
      %ѵ����    
%       data_1 =zeros(150,1); %12-2
%       data_2 =zeros(153,1); %3-5
%       data_3 =zeros(153,1); %6-8
%       data_4 =zeros(152,1); %9-11
%       
%        a=0;
%        b=0;
%        c=0;
%        d=0;
%        
%        for i=1:1:59
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%        end
% 
%        for i=60:1:151
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=152:1:243
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=244:1:334
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end
%        
%         for i=335:1:425
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end
%         
%         for i=426:1:486
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
%         
%         
%         for i=518:1:578
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%         end
%         
%         
%          for i=610:1:670
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%          end
        
         
%          %���Լ�
      data_1 =zeros(31,1); %12-2
      data_2 =zeros(31,1); %3-5
      data_3 =zeros(31,1); %6-8
      data_4 =zeros(30,1); %9-11
      
       a=0;
       b=0;
       c=0;
       d=0;
       
       for i=701:1:731
           for j=1:1:1
               a=a+1;
               data_1(a,1)=data(i,j);
           end
       end

       for i=487:1:517
           for j=1:1:1
               b=b+1;
               data_2(b,1)=data(i,j);
           end
        end

        for i=579:1:609
           for j=1:1:1
               c=c+1;
               data_3(c,1)=data(i,j);
           end
         end

        for i=671:1:700
           for j=1:1:1
               d=d+1;
               data_4(d,1)=data(i,j);
           end
        end
%        


% ��������
%       data_1 =zeros(181,1); %12-2
%       data_2 =zeros(184,1); %3-5
%       data_3 =zeros(184,1); %6-8
%       data_4 =zeros(182,1); %9-11
%       
%        a=0;
%        b=0;
%        c=0;
%        d=0;
%        
%        for i=1:1:59
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%        end
% 
%        for i=60:1:151
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=152:1:243
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=244:1:334
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end
%        
%         for i=335:1:425
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end
%         
%         for i=426:1:517
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
%         
%         
%         for i=518:1:609
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%         end
%         
%         
%          for i=610:1:700
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%          end
%          
%          for i=701:1:731
%            for j=1:1:1
%                a=a+1;
%                data_1(d,1)=data(i,j);
%            end
%          end
%          
        SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��

        outfile1=strcat(SaveFolder1,'\',SaveFiles);
        outfile2=strcat(SaveFolder2,'\',SaveFiles);
        outfile3=strcat(SaveFolder3,'\',SaveFiles);
        outfile4=strcat(SaveFolder4,'\',SaveFiles);

        
        fid1=fopen(outfile1,'w');
        fid2=fopen(outfile2,'w');
        fid3=fopen(outfile3,'w');
        fid4=fopen(outfile4,'w');
        

         for i=1:1:31
             for j=1:1:1
                 if j==1
                     fprintf(fid1,'%g\r\n',data_1(i,j));
                 else
                    fprintf(fid1,'%g ',data_1(i,j));
                 end
             end   
         end
         fclose(fid1); 
         
         for i=1:1:31
             for j=1:1:1
                 if j==1
                     fprintf(fid2,'%g\r\n',data_2(i,j));
                 else
                    fprintf(fid2,'%g ',data_2(i,j));
                 end
             end   
         end   
         fclose(fid2);
         
         for i=1:1:31
             for j=1:1:1
                 if j==1
                     fprintf(fid3,'%g\r\n',data_3(i,j));
                 else
                    fprintf(fid3,'%g ',data_3(i,j));
                 end
             end   
         end      
         fclose(fid3); 
         
         for i=1:1:30
             for j=1:1:1
                 if j==1
                     fprintf(fid4,'%g\r\n',data_4(i,j));
                 else
                    fprintf(fid4,'%g ',data_4(i,j));
                 end
             end   
         end         
         
         fclose(fid4); 
               
end
disp('�������')



%17��
% parfor k=3:FilesCount
%       FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
%       Name=Files(k).name;
%       location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��
%   
% %       i= str2num(Name(location(end)-6:location(end)-4));
% %       j= str2num(Name(location(end)-3:location(end)-1));
% 
%       fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data = cell2mat(textscan(fid,'%f','headerlines',0));
%       data = reshape(data,1,365);
%       data = data'; 
%       fclose(fid);
%       
%       data_1 =zeros(90,1); %12-2
%       data_2 =zeros(92,1); %3-5
%       data_3 =zeros(92,1); %6-8
%       data_4 =zeros(91,1); %9-11
%       
%        a=0;
%        b=0;
%        c=0;
%        d=0;
%        
%        for i=1:1:59
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%        end
% 
%        for i=60:1:151
%            for j=1:1:1
%                b=b+1;
%                data_2(b,1)=data(i,j);
%            end
%         end
% 
%         for i=152:1:243
%            for j=1:1:1
%                c=c+1;
%                data_3(c,1)=data(i,j);
%            end
%          end
% 
%         for i=244:1:334
%            for j=1:1:1
%                d=d+1;
%                data_4(d,1)=data(i,j);
%            end
%         end
%        
%         for i=335:1:365
%            for j=1:1:1
%                a=a+1;
%                data_1(a,1)=data(i,j);
%            end
%         end
%         
%          
%         SaveFiles=strcat(Name(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��
% 
%         outfile1=strcat(SaveFolder1,'\',SaveFiles);
%         outfile2=strcat(SaveFolder2,'\',SaveFiles);
%         outfile3=strcat(SaveFolder3,'\',SaveFiles);
%         outfile4=strcat(SaveFolder4,'\',SaveFiles);
% 
%         
%         fid1=fopen(outfile1,'w');
%         fid2=fopen(outfile2,'w');
%         fid3=fopen(outfile3,'w');
%         fid4=fopen(outfile4,'w');
%         
% 
%          for i=1:1:90
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid1,'%g\r\n',data_1(i,j));
%                  else
%                     fprintf(fid1,'%g ',data_1(i,j));
%                  end
%              end   
%          end
%          fclose(fid1); 
%          
%          for i=1:1:92
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid2,'%g\r\n',data_2(i,j));
%                  else
%                     fprintf(fid2,'%g ',data_2(i,j));
%                  end
%              end   
%          end   
%          fclose(fid2);
%          
%          for i=1:1:92
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid3,'%g\r\n',data_3(i,j));
%                  else
%                     fprintf(fid3,'%g ',data_3(i,j));
%                  end
%              end   
%          end      
%          fclose(fid3); 
%          
%          for i=1:1:91
%              for j=1:1:1
%                  if j==1
%                      fprintf(fid4,'%g\r\n',data_4(i,j));
%                  else
%                     fprintf(fid4,'%g ',data_4(i,j));
%                  end
%              end   
%          end         
%          
%          fclose(fid4); 
%                
% end
% disp('�������')