import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.JapaneseTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

import java.io.File;
import java.io.FileNotFoundException;

public class SimpleWord2vec {
    private File file;
    // テキストファイルなどを一行ずつ読むためのイテレータです。
    private LineSentenceIterator iterator;
    private TokenizerFactory tokenizerFactory;

    private Word2Vec word2Vec;

    public SimpleWord2vec(File file, TokenizerFactory tokenizerFactory) throws FileNotFoundException {
        this.file = file;
        this.iterator = new LineSentenceIterator(this.file);
        this.tokenizerFactory = tokenizerFactory;
        this.tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }

    public void init() {
        // モデルのビルドをします。
        this.word2Vec = new Word2Vec.Builder()
                .minWordFrequency(1)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(this.iterator)
                .tokenizerFactory(this.tokenizerFactory)
                .build();
    }

    public void train() {
        this.word2Vec.fit();
    }


    public static void main(String... args) throws FileNotFoundException {
        // 日本語の文章に対してWord2Vecを適用します。
        SimpleWord2vec word2vec = new SimpleWord2vec(
                new File("resources/sample.txt"),
                new JapaneseTokenizerFactory()
        );
        word2vec.init();
        word2vec.train();

        word2vec.word2Vec.wordsNearest("寿司",5)
                .forEach(str -> {
                    System.out.print("単語間類似度："+ "寿司<=>" + str + " ");
                    System.out.println(word2vec.word2Vec.similarity("寿司",str));
                });
        // この例ではデータ数が足りていないので、あまり良い結果が得られていません。
        // 詳しいWord2Vecのチューニングは、下のページを見てください。
        // https://deeplearning4j.org/docs/latest/deeplearning4j-nlp-word2vec
        return ;
    }
}
