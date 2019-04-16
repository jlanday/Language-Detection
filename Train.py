def main(args):
    Now = datetime.datetime.now().strftime('%Y-%m-%d')
    ModelName = '{}_{}_{}_{}_{}_{}_{}_{}'.format(args.ModelType, args.WordOrChar, Now , args.batch_size, args.sequence_length, args.Embedding_Dim, args.Neurons, args.DropOutRate)
    # -- Download the data from GCS.
    logger.info('Loading Data ...')
    X, Y, Language2Indx, Indx2Language = LoadData(args.N_Rows)
    # -- Create Lexicons.
    logger.info('Making Vocabularly ...')
    Word2Indx, Indx2Word = MakeVocabulary(X, args.WordOrChar)
    # -- Upload Lexicons to GCS.
    logger.info('Uploading Lexicons ...')
    UploadBlob('lexicons', 'Word2Indx.json', ModelName)
    UploadBlob('lexicons', 'Indx2Word.json', ModelName)
    UploadBlob('lexicons', 'Language2Indx.json', ModelName)
    UploadBlob('lexicons', 'Indx2Language.json', ModelName)
    # -- Convert the strings to numerical representation.
    logger.info('Creating Matricies ...')
    X = CreateMatricies(X, Word2Indx, args.sequence_length, args.WordOrChar)
    # -- Create train/test split.
    logger.info('Creating Test / Train split ...')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.0002, random_state = 666)
    # -- Make test set the size of one batch
    if len(X_test) >= args.batch_size:
        X_test, Y_test = X_test[0:args.batch_size], Y_test[0:args.batch_size]

    # -- Load the model.
    if args.ModelType == 'FullyConnected':
        model = FullyConnected(Word2Indx, Language2Indx, args.sequence_length, args.Embedding_Dim, args.Neurons, args.DropOutRate)
    elif args.ModelType == 'TwoLSTM':
        model = TwoLSTM(Word2Indx, Language2Indx, args.sequence_length, args.Embedding_Dim, args.Neurons, args.DropOutRate)
    elif args.ModelType == 'OneBidirectionalLSTM':
        model = OneBidirectionalLSTM(Word2Indx, Language2Indx, args.sequence_length, args.Embedding_Dim, args.Neurons, args.DropOutRate)

    # -- Define model callbacks.
    callback = TensorBoard(log_dir = 'gs://XXX/models/nlp/language_classification/{}/logs'.format(ModelName),
                                            histogram_freq = 0, batch_size = args.batch_size , write_graph = False,
                                            write_images = False, embeddings_freq = 0, embeddings_layer_names = None,
                                            embeddings_metadata = None, embeddings_data = None, update_freq = len(X_train)//50
                                            )
    # -- Train the model.
    logger.info('Fitting Model ...')
    model.fit(x = X_train, y = Y_train, batch_size = args.batch_size, epochs = args.N_Epochs, verbose = 1,
              validation_data = (X_test, Y_test), callbacks = [callback])
    # -- Save the model weights.
    logger.info('Saving Model Weights...')
    model.save_weights('Weights.h5')
    logger.info('Uploading Model Weights...')
    UploadBlob('model_weights','Weights.h5', ModelName)

if __name__ == '__main__':
    from google.cloud import storage
    from google.cloud.storage import Blob
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, CuDNNLSTM
    from keras.utils import to_categorical
    from keras.callbacks import TensorBoard
    from keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split
    import random,os,json,datetime,logging,argparse

    def LoadData(NRows):
        logger = logging.getLogger('Language_Classification')
        '''Loads all of the training data from google cloud storage into local memory.

        Args:
            NRows (int) - The maximum number of rows per language.
        Returns
            X (list) - A list of sentences.
            Y (list) - The language of the sentence.
            Language2Indx (dict) - A one-to-one mapping between language and a numerical index.
            Indx2Language (dict) - The inverted version of above.
        '''
        # -- Initialize some variables.
        Language2Indx = {}
        X,Y = [],[]

        client = storage.Client()
        bucket = client.bucket('XXX')
        # -- Index is used to keep a numerical representation of the language.
        indx = 0
        for blob in bucket.list_blobs(prefix='data/nlp/language_classification'):
            try:
                # -- Language name is in the blobs name.
                name = blob.name.split('/')[-1]
                if name != '':
                    lang = name.split('.')[0]
                    logger.info('Downloading the data for {}.'.format(lang))
                    tempdata = blob.download_as_string().decode().split('\n')
                    tempdata = [row for row in tempdata if row.strip()!='']
                    # -- Only take the languages with a ton of training data.
                    if len(tempdata)>= NRows:
                    # -- Only take the first NRows rows.
                        if len(tempdata) <= NRows:
                            Nrows = len(tempdata)
                        else:
                            Nrows = NRows
                        for i in range(Nrows):
                            X.append(tempdata[i].strip())
                            Y.append(indx)
                        indx = indx + 1
                        logger.info('{} : {} '.format(lang,len(Y)))
                        Language2Indx[lang] = indx
                    else:
                        logger.info('Did not download {}. It only has {} rows.'.format(lang,len(tempdata)))

            except Exception as e:
                logger.info(e)
        # -- Conver the numerical index to a vector representation of Kronecker Delta.
        # IE 1 --> [1 , 0 , 0, ..., 0 ], 2 -> [0, 1, 0 , ..., 0 ], N --> [0, 0, 0, ..., 1]
        Y = to_categorical(Y)
        # -- Invert the mapping
        Indx2Language = InvertDict(Language2Indx)
        # -- Save those mappings locally so they can be uploaded to GCS downstream.
        with open('Language2Indx.json','w') as f:
            f.write(json.dumps(Language2Indx))
        with open('Indx2Language.json','w') as f:
            f.write(json.dumps(Indx2Language))
        return X,Y,Language2Indx,Indx2Language

    def InvertDict(d):
        logger = logging.getLogger('Language_Classification')
        '''A helper function to invert a dictionary.
        '''
        assert type(d) == dict
        return {vv:kk for kk,vv in d.items()}

    def MakeVocabulary(data,WordOrChar):
        logger = logging.getLogger('Language_Classification')
        '''Creates a vocabulary (lexicon) for our source Words (or characters) and mapps them to a numerical index.

        Args:
            data (list) - A list of strings.
            WordOrChar (string) - Do you want to do a word based model or a character based model? Use "word"
            here for words, Anything else for character.

        Returns:
            Word2Indx (dict) - A mapping between words (or characters) to a numerical index
            Indx2Word (dict) - An inverse of the above.


        '''
        #-- Initize the lexicon with padding tokens, starts, stops, and unknowns.
        Word2Indx = {'<padding>' : 0, '<start>' : 1 , '<end>' : 2 , '<unk>' : 3 }
        WordFreq = {}
        for indx, row in enumerate(data):
            if indx % 55000 == 0:
                logger.info('We are {:.3f} percent done making WordFreq'.format(indx/len(data)))

            # -- Split based on whether we are doing a word or character based model.
            if WordOrChar.lower() == 'word':
                row = row.strip().split()
            else:
                row = row.strip()

            # -- Collect word frequencies for each word
            for word in row:
                word = word.lower()
                if word not in WordFreq:
                    WordFreq[word] = 1
                else:
                    WordFreq[word] = WordFreq[word] + 1

        for indx, row in enumerate(data):
            if indx % 55000 == 0:
                logger.info('We are {:.3f} percent done making Word2Indx'.format(indx/len(data)))

            # -- Split based on whether we are doing a word or character based model.
            if WordOrChar.lower() == 'word':
                row = row.strip().split()
            else:
                row = row.strip()

            # -- Only keep words if they appear 5 or more times and characters if they appear more than once.
            if WordOrChar == 'word':
                filter = 5
            else:
                filter = 0
            for word in row:
                word = word.lower()
                if word in WordFreq:
                    if WordFreq[word] >= filter and word not in Word2Indx:
                        Word2Indx[word] = len(Word2Indx)

        logger.info('There are {} words.'.format(len(WordFreq)))
        logger.info('There are {} words that appear more than once.'.format(len(Word2Indx)))

        Indx2Word = InvertDict(Word2Indx)
        # -- Write the dicts to local memory so they can be later uploaded to GCS.
        with open('Word2Indx.json','w') as f:
            f.write(json.dumps(Word2Indx))
        with open('Indx2Word.json','w') as f:
            f.write(json.dumps(Indx2Word))
        return Word2Indx,Indx2Word

    def CreateMatricies(data, Word2Indx, sequence_length, WordOrChar):
        logger = logging.getLogger('Language_Classification')
        '''Creates a numerical representation (matrix) for our source words (or characters).

        Args:
            data (list) - A list of strings.
            Word2Indx (dict) - A mapping between words (or characters) to a numerical index.
            sequence_length (int) - How many tokens in a sequence?
            WordOrChar (string) - Are we doing a word based or character based model?


        Returns:
            X (matrix) - A matrix containing the data needed by Keras/TF.
        '''
        X = []
        for indx, row in enumerate(data):
            if indx % 55000 == 0:
                logger.info('We are {:.3f} percent done creating a matrix.'.format(indx/len(data)))
            # -- All sentences have the same start token.
            sentence = [1]
            # -- Tokenize by words or characters
            if WordOrChar.lower() == 'word':
                row = row.strip().split()
            else:
                row = row.strip()
            # -- Create a vector per row.
            for word in row:
                word = word.lower()
                if word in Word2Indx:
                    sentence.append(Word2Indx[word])
                else:
                    sentence.append(Word2Indx['<unk>'])
            # -- All sentences end with same end token.
            sentence.append(Word2Indx['<end>'])
            X.append(sentence)
        # -- Pad all sequences to the same length.
        X = pad_sequences(X,sequence_length,padding='post',truncating='post')
        # -- Since we are padding, all sentences longer than sequence length won't have the correct end token. This loop fixes that.
        for row_idx, row in enumerate(X):
            if 2 not in row:
                X[row_idx][-1] = 2
        return X

    def UploadBlob(FileType, FileName, ModelName):
        logger = logging.getLogger('Language_Classification')
        '''Uploads a blob to GCS from a local file.
        '''
        blob = Blob('models/nlp/language_classification/{}/{}/{}'.format(ModelName,FileType,FileName),bucket)
        blob.upload_from_filename(FileName)


    def OneBidirectionalLSTM(Word2Indx, Language2Indx, sequence_length, Embedding_Dim, Neurons, DropOutRate):
        model = Sequential()
        model.add(Embedding(len(Word2Indx), Embedding_Dim, input_length = sequence_length))
        model.add(Bidirectional(LSTM(Neurons)))
        model.add(Dropout(DropOutRate))
        model.add(Dense(len(Language2Indx), activation='softmax'))
        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
        return model


    def TwoLSTM(Word2Indx, Language2Indx, sequence_length, Embedding_Dim, Neurons, DropOutRate):
        model = Sequential()
        model.add(Embedding(len(Word2Indx), Embedding_Dim, input_length = sequence_length))
        model.add(LSTM(Neurons,return_sequences = True, dropout = DropOutRate, recurrent_dropout = DropOutRate))
        model.add(LSTM(Neurons, dropout = DropOutRate, recurrent_dropout = DropOutRate))
        model.add(Dense(len(Language2Indx), activation='softmax'))
        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
        return model

    def FullyConnected(Word2Indx, Language2Indx, sequence_length, Embedding_Dim, Neurons, DropOutRate):
        model = Sequential()
        # model.add(Embedding(len(Word2Indx), Embedding_Dim, input_length = sequence_length))
        model.add(Dense(500, kernel_initializer="glorot_uniform", activation="sigmoid"))
        model.add(Dropout(DropOutRate))
        model.add(Dense(300, kernel_initializer="glorot_uniform", activation="sigmoid"))
        model.add(Dropout(DropOutRate))
        model.add(Dense(100, kernel_initializer="glorot_uniform", activation="sigmoid"))
        model.add(Dropout(DropOutRate))
        model.add(Dense(len(Language2Indx), activation='softmax'))
        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
        return model



    client = storage.Client()
    bucket = client.bucket('XXX')

    logger = logging.getLogger('Language_Classification')
    logger.setLevel(logging.INFO)
    info_handler = logging.StreamHandler()
    info_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s',datefmt = '%m/%d/%Y %I:%M:%S %p')
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)

    parser = argparse.ArgumentParser(description = "All hail the great satan")
    parser.add_argument('--ModelType', '-m', required = True, action = 'store', help = 'FullyConnected, TwoLSTM, OneBidirectionalLSTM', type = str)
    parser.add_argument('--WordOrChar', '-wc', required = True, action = 'store', help = 'Do you want a word or character based model?', type = str)
    parser.add_argument('--N_Rows', '-nr', action = 'store', help = 'How many rows per language?', default = 30000, type = int)
    parser.add_argument('--N_Epochs', '-ne', action = 'store' ,help = 'How many epochs do you want to train?', default = '3', type = int)
    parser.add_argument('--batch_size', '-bs', action = 'store', help = 'How many samples do you want in your batch?', default = '256', type = int)
    parser.add_argument('--sequence_length', '-sl', action = 'store', help = 'How many tokens are in a sequence?', default = '10', type = int)
    parser.add_argument('--Embedding_Dim', '-ed', action = 'store', help = 'How many dimensions should each word vector have?', default = '256', type = int)
    parser.add_argument('--Neurons', '-n', action = 'store', help = 'How many cells in LSTM?', default = '1024', type = int)
    parser.add_argument('--DropOutRate', '-d', action = 'store', help = 'How many cells in LSTM?', default = '0.2', type = float)

    args = parser.parse_args()

    main(args)
