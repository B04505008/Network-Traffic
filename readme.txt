

Data Preparing:
  1.Put pcap files in folder 'pcaps'
  2.Run pcap_to_Data(flow).ipynb
  3.It can make a list of 2d-arrays(flows) from a pcap (ex: youtube1.pcap -> youtube1.pickle and youtube1_class.pickle)

Make Training Data:
  1.Run this command:
    python3 main.py -m ccccc
  It can generate datas for training and validation (like X_train.pickle , X_val.pickle ,y_train.pickle ...)
  
Machine Learining:
  Pytorch should have already been installed
  https://pytorch.org/get-started/locally/

  1.Run this command:
    python3 torchmain.py
  2.After training, run predict.ipynb. It can predict datas by saved models and generate the confusion matrix(csv file)
  3.Run get_acc.ipynb,it can read a csv array file and plot the confusion matrix image.
  