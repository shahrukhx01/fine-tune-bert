# Fine Tune Bert for Text Classification

Fine tuning Google's Bert using transformers package and Pytorch on the following dataset:
SMS Spam Collection v. 1
The SMS Spam Collection v.1 is a public set of SMS labeled messages that have been collected for mobile phone spam research. It has one collection composed by 5,574 English, real and non-enconded messages, tagged according being legitimate (ham) or spam.

Composition
This corpus has been collected from free or free for research sources at the Internet:

A collection of 425 SMS spam messages was manually extracted from the Grumbletext Web site. This is a UK forum in which cell phone users make public claims about SMS spam messages, most of them without reporting the very spam message received. The identification of the text of spam messages in the claims is a very hard and time-consuming task, and it involved carefully scanning hundreds of web pages. The Grumbletext Web site is: http://www.grumbletext.co.uk/.
A subset of 3,375 SMS randomly chosen ham messages of the NUS SMS Corpus (NSC), which is a dataset of about 10,000 legitimate messages collected for research at the Department of Computer Science at the National University of Singapore. The messages largely originate from Singaporeans and mostly from students attending the University. These messages were collected from volunteers who were made aware that their contributions were going to be made publicly available. The NUS SMS Corpus is avalaible at: http://www.comp.nus.edu.sg/~rpnlpir/downloads/corpora/smsCorpus/.
A list of 450 SMS ham messages collected from Caroline Tag's PhD Thesis available at http://etheses.bham.ac.uk/253/1/Tagg09PhD.pdf.
Finally, we have incorporated the SMS Spam Corpus v.0.1 Big. It has 1,002 SMS ham messages and 322 spam messages and it is public available at: http://www.esp.uem.es/jmgomez/smsspamcorpus/.
The table below lists the provided dataset in different file formats, the amount of samples in each class and the total number of samples.

Application	File format	# Spam	# Ham	Total	Link
General	Plain text	747	4,827	5,574	Link 1
Weka	ARFF	747	4,827	5,574	Link 2Note: the messages are not in a chronological order.

You can find more useful information about the SMS Spam Collection v.1 at the following page of the UCI Repository.
Usage
The collection is composed by just one file, where each line has the correct class (ham or spam) followed by the raw message.


ham   What you doing?how are you?
ham   Ok lar... Joking wif u oni...
ham   dun say so early hor... U c already then say...
ham   MY NO. IN LUTON 0125698789 RING ME IF UR AROUND! H*
ham   Siva is in hostel aha:-.
ham   Cos i was out shopping wif darren jus now n i called him 2 ask wat present he wan lor. Then he started guessing who i was wif n he finally guessed darren lor.
spam  FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop
spam  Sunshine Quiz! Win a super Sony DVD recorder if you canname the capital of Australia? Text MQUIZ to 82277. B
spam  URGENT! Your Mobile No 07808726822 was awarded a L2,000 Bonus Caller Prize on 02/09/03! This is our 2nd attempt to contact YOU! Call 0871-872-9758 BOX95QU


We would appreciate:

If you find this collection useful, make a reference to the paper below and the web page: http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/
Send us a message either to talmeida < AT > ufscar.br  or jmgomezh <AT> yahoo.es in case you make use of the corpus.
Publication and More Information
We offer a comprehensive study of this corpus in the following papers. These works present a number of interesting statistics, studies and baseline results for many traditional machine learning methods.

Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. Contributions to the Study of SMS Spam Filtering: New Collection and Results.  Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11), Mountain View, CA, USA, 2011. [preprint]

Gómez Hidalgo, J.M., Almeida, T.A., Yamakami, A. On the Validity of a New SMS Spam Collection.  Proceedings of the 11th IEEE International Conference on Machine Learning and Applications (ICMLA'12), Boca Raton, FL, USA, 2012. [preprint]

Almeida, T.A., Gómez Hidalgo, J.M., Silva, T.P.  Towards SMS Spam Filtering: Results under a New Dataset.   International Journal of Information Security Science (IJISS), 2(1), 1-18, 2013. [Invited paper - full version]

About
The SMS Spam Collection has been created by Tiago A. Almeida and José María Gómez Hidalgo.

We would like to thank Min-Yen Kan and his team for making the NUS SMS Corpus available.


© Tiago A. Almeida and José María Gómez Hidalgo, 2011.
