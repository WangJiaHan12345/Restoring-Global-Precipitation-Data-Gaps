%%����֮ǰ�ܳ����������������ݵõ���������Ҫ������
%%xunlian
% SaveFolder=strcat('H:\ʱ��Ԥ��\�ĸ������������\ganhan\02_grid_data\xunlian\','final'); %����ļ���·��
% if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
%     mkdir(SaveFolder);
% end
% disp('������...');
% 
% FolderPath1=input('���������ݴ洢�ļ���:','s'); %����cpc early final��01_clip_data 12-2  ʵ�����õ��ǿռ�Ԥ�������02_final_data
% Files1=dir(FolderPath1);
% FilesCount1=length(Files1);
% 
% FolderPath2=input('���������ݴ洢�ļ���:','s'); %���� 3-5
% Files2=dir(FolderPath2);
% FilesCount2=length(Files2);
% 
% FolderPath3=input('���������ݴ洢�ļ���:','s'); %���� 6-8
% Files3=dir(FolderPath3);
% FilesCount3=length(Files3);
% 
% FolderPath4=input('���������ݴ洢�ļ���:','s'); %����  9-11
% Files4=dir(FolderPath4);
% FilesCount4=length(Files4);
% 
% 
% parfor k=3:FilesCount1 
%       data=zeros(1096,1);
%       FilePath1=strcat(FolderPath1,'\',Files1(k).name);  %�ļ�·��\�ļ���
%       Name1=Files1(k).name;
%       location=strfind(Name1,'.');  %����ַ�'.'��FilePath��λ��
%       
%       FilePath2=strcat(FolderPath2,'\',Files2(k).name);  %�ļ�·��\�ļ���
%       Name2=Files2(k).name;
%      
%       
%       FilePath3=strcat(FolderPath3,'\',Files3(k).name);  %�ļ�·��\�ļ���
%       Name3=Files3(k).name;
%       
%       
%       FilePath4=strcat(FolderPath4,'\',Files4(k).name);  %�ļ�·��\�ļ���
%       Name4=Files4(k).name;
%      
%   
%       fid1=fopen(FilePath1,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data1 = cell2mat(textscan(fid1,'%f','headerlines',0));
%       data1 = reshape(data1,1,361);
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
%       data3 = reshape(data3,1,368);
%       data3 = data3';
%       fclose(fid3); 
%       
%       fid4=fopen(FilePath4,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
%       data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
%       data4 = reshape(data4,1,364);
%       data4 = data4';
%       fclose(fid4); 
%       
%       data(1:59,1)=data1(1:59,1); 
%       data(60:151,1)=data2(1:92,1); 
%       data(152:243,1)=data3(1:92,1); 
%       data(244:334,1)=data4(1:91,1); 
%       data(335:425,1)=data1(60:150,1); 
%       data(426:517,1)=data2(93:184,1); 
%       data(518:609,1)=data3(93:184,1);
%       data(610:700,1)=data4(92:182,1); 
%       data(701:790,1)=data1(151:240,1); 
%       data(791:882,1)=data2(185:276,1); 
%       data(883:974,1)=data3(185:276,1); 
%       data(975:1065,1)=data4(183:273,1); 
%       data(1066:1096,1)=data1(241:271,1); 
%       
%            
%      SaveFiles=strcat(Name1(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��
%      SaveFiles=strcat(Name(1:location(end)-1),'.txt');
%     
%      outfile=strcat(SaveFolder,'\',SaveFiles);
% 
%      if exist(outfile,'file')~=0 
%         delete(outfile);     
%      end
%      fid5=fopen(outfile,'w');
%      
% 
%      for i=1:1:1096
%          for j=1:1:1
%              if j==1
%                  fprintf(fid5,'%g\r\n',data(i,j));
%              else
%                 fprintf(fid5,'%g ',data(i,j));
%              end
%          end   
%      end
%      fclose(fid5); 
%    
% 
% end
% disp('�������')



% % %ceshi
SaveFolder=strcat('H:\ʱ��Ԥ��\�ĸ������������\ganhan\02_grid_data\ceshi\','cpc'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end
disp('������...');

FolderPath1=input('���������ݴ洢�ļ���:','s'); %����
Files1=dir(FolderPath1);
FilesCount1=length(Files1);

FolderPath2=input('���������ݴ洢�ļ���:','s'); %����
Files2=dir(FolderPath2);
FilesCount2=length(Files2);

FolderPath3=input('���������ݴ洢�ļ���:','s'); %����
Files3=dir(FolderPath3);
FilesCount3=length(Files3);

FolderPath4=input('���������ݴ洢�ļ���:','s'); %����
Files4=dir(FolderPath4);
FilesCount4=length(Files4);


parfor k=3:FilesCount1 
      data=zeros(365,1);
      FilePath1=strcat(FolderPath1,'\',Files1(k).name);  %�ļ�·��\�ļ���
      Name1=Files1(k).name;
      location=strfind(Name1,'.');  %����ַ�'.'��FilePath��λ��
      
      FilePath2=strcat(FolderPath2,'\',Files2(k).name);  %�ļ�·��\�ļ���
      Name2=Files2(k).name;
     
      
      FilePath3=strcat(FolderPath3,'\',Files3(k).name);  %�ļ�·��\�ļ���
      Name3=Files3(k).name;
      
      
      FilePath4=strcat(FolderPath4,'\',Files4(k).name);  %�ļ�·��\�ļ���
      Name4=Files4(k).name;
     
  
      fid1=fopen(FilePath1,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data1 = cell2mat(textscan(fid1,'%f','headerlines',0));
      data1 = reshape(data1,1,361);
      data1 = data1';
      fclose(fid1); 
      
      fid2=fopen(FilePath2,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data2 = cell2mat(textscan(fid2,'%f','headerlines',0));
      data2 = reshape(data2,1,368);
      data2 = data2';
      fclose(fid2); 
      
      fid3=fopen(FilePath3,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data3 = cell2mat(textscan(fid3,'%f','headerlines',0));
      data3 = reshape(data3,1,368);
      data3 = data3';
      fclose(fid3); 
      
      fid4=fopen(FilePath4,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
      data4 = cell2mat(textscan(fid4,'%f','headerlines',0));
      data4 = reshape(data4,1,364);
      data4 = data4';
      fclose(fid4); 
      
      data(1:59,1)=data1(272:330,1); 
      data(60:151,1)=data2(277:368,1); 
      data(152:243,1)=data3(277:368,1); 
      data(244:334,1)=data4(274:364,1); 
      data(335:365,1)=data1(331:361,1); 
    
      
           
     SaveFiles=strcat(Name1(location(end)-6:location(end)-1),'.txt'); %CPC����ļ���·��
     %SaveFiles=strcat(Name(1:location(end)-1),'.txt');
    
     outfile=strcat(SaveFolder,'\',SaveFiles);

     if exist(outfile,'file')~=0 
        delete(outfile);     
     end
     fid5=fopen(outfile,'w');
     

     for i=1:1:365
         for j=1:1:1
             if j==1
                 fprintf(fid5,'%g\r\n',data(i,j));
             else
                fprintf(fid5,'%g ',data(i,j));
             end
         end   
     end
     fclose(fid5); 
   

end
disp('�������')
