FolderPath=input('���������ݴ洢�ļ���:','s'); %����
index=strfind(FolderPath,'\');  %����ַ�'\'��FolderPath��λ��
SaveFolder=strcat('H:\ʱ��Ԥ��\������\24��\02_grid_data\Early\','xunlian'); %����ļ���·��
if exist(SaveFolder,'dir')~=7  %���·�����������½�·��
    mkdir(SaveFolder);
end
Files=dir(FolderPath);
FilesCount=length(Files);
disp('������...');

grid_count=562;
for i=1:1:562
    data2=zeros(365,1); 
    a=0;
    for k=3:FilesCount
          a=a+1;
          FilePath=strcat(FolderPath,'\',Files(k).name);  %�ļ�·��\�ļ���
          Name=Files(k).name;
          location=strfind(Name,'.');  %����ַ�'.'��FilePath��λ��

          fid=fopen(FilePath,'rb','l');  % 'rb'�Զ����Ʒ�ʽֻ�����ʹ��ļ���Ҳ����ֱ��'r';'l':little endianС�����
          data = cell2mat(textscan(fid,'%f','headerlines',0));
          data = reshape(data,1,562);
          data = data';
          fclose(fid); 
          
          data2(a,1)=data(i,1);     
    end
    
    SaveFiles=strcat(num2str(i),'.txt'); %CPC����ļ���·��
    outfile=strcat(SaveFolder,'\',SaveFiles);

    if exist(outfile,'file')~=0 
    delete(outfile);     
    end
    fid1=fopen(outfile,'w');


     for i=1:1:365
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


 