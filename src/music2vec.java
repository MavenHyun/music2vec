/**
 * Created by Maven Hyun on 2017-05-25.
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;
import com.aliasi.tokenizer.*;
import com.aliasi.tokenizer.TokenizerFactory;
import Jama.Matrix;
import org.jsoup.*;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;


public class music2vec
{
    public int N = 0; /*number of features*/
    public int V = 0; /*vocabulary size*/
    public int C = 0; /*window size*/
    public int G = 0;
    public double L = 0.0; /*learning rate*/
    public TokenizerFactory factory; /*scv constructs terran factory*/
    public int total_V = 0; /*vocabulary size including duplicates*/

    public Map<String, Integer> vocab_table = new HashMap<String, Integer>();
    public Map<String, String> song_table = new HashMap<String, String>();
    public Map<String, Integer> vocab_num = new HashMap<String, Integer>();
    public Map<String, Double> vocab_neg = new HashMap<String, Double>();
    public ArrayList<String> song_lyrics = new ArrayList<String>();
    public ArrayList<String> context_words = new ArrayList<String>();
    public ArrayList<String> negative_words = new ArrayList<String>();

    public Set<String> stopwords = new HashSet<String>();

    public Matrix input2hidden_weight = Matrix.random(V, N);
    public Matrix hidden2output_weight = Matrix.random(N, V);

    public Matrix input = new Matrix(V,1);
    public Matrix hidden = new Matrix(N,1);
    public Matrix output = new Matrix(V,1);
    public Matrix error = new Matrix(V,1);

    music2vec(int context, int feature, double learn, int negative)
    {
        N = feature;
        C = context;
        L = learn;
        G = negative;
        try {
            FileReader f = new FileReader("stopword.txt");
            BufferedReader b = new BufferedReader(f);
            String line = null;
            while ((line = b.readLine()) != null) {
                stopwords.add(line);
            }
        } catch (Exception e) {
        }
    }

    public void song_collect()
    {
        try {
            FileReader f = new FileReader("songlist.txt");
            BufferedReader b = new BufferedReader(f);
            String line = null;
            while ((line = b.readLine()) != null)
            {
                String[] text = line.split("\t");
                String text01 = text[0] + " | " + text[1];
                song_lyrics.add(text[2]);
                song_table.put(text[2], text01);
                vocab_table.put(text01, vocab_table.size());
            }
        } catch(Exception e){}
    }

    public void song_tokenize()
    {
        factory = new IndoEuropeanTokenizerFactory();
        LowerCaseTokenizerFactory low = new LowerCaseTokenizerFactory(factory);
        EnglishStopTokenizerFactory stop = new EnglishStopTokenizerFactory(factory);
        PorterStemmerTokenizerFactory stem = new PorterStemmerTokenizerFactory(factory);

        for (String lyrics : song_lyrics)
        {
            Tokenization tokens = new Tokenization(lyrics, factory);
            for (String token : tokens.tokens())
            {
                try {
                    total_V++;
                    String target = stem.modifyToken(stop.modifyToken(low.modifyToken(token)));
                    if (!vocab_table.containsKey(target)) vocab_table.put(target, vocab_table.size());
                    if (!vocab_num.containsKey(target)) vocab_num.put(target, 1);
                    else vocab_num.replace(target, vocab_num.get(target), vocab_num.get(target) + 1);
                } catch (Exception e) { }
            }
        }

        V = vocab_table.size();
        input2hidden_weight = Matrix.random(V, N);
        hidden2output_weight = Matrix.random(N, V);
    }

    public void song_negative()
    {
        double sum = 0.0;
        for (String word : vocab_num.keySet()) sum += Math.pow((double) vocab_num.get(word), 0.75);
        for (String word: vocab_num.keySet())
        {
            double prob = (double) vocab_num.get(word) / sum;
            vocab_neg.put(word, prob);
        }
    }

    public Matrix vectorize(String word)
    {
        double[] array = new double[V];
        array[vocab_table.get(word)] = 1;
        Matrix vector = new Matrix(array, 1);
        return vector.transpose();
    }

    public Matrix input_vector() /*project all context words(part of lyrics) through i2h weight matrix*/
    {
        Matrix vector = new Matrix(V, 1);
        for (String word : context_words) vector.plusEquals(vectorize(word));
        return vector;
    }

    public Matrix hidden_vector()
    {
        Matrix vector = new Matrix(N, 1);
        vector = (input2hidden_weight.transpose()).times(input_vector());
        vector = vector.times(1/(double)C);
        hidden = vector;
        return vector;
    }

    public Matrix output_vector()
    {
        Matrix vector = new Matrix(V, 1);
        vector = (hidden2output_weight.transpose()).times(hidden_vector());
        double sum = 0;
        for (int i = 0; i < V; i++)
        {
            vector.set(i, 0, Math.exp(vector.get(i, 0)));
            sum += vector.get(i, 0);
        }
        output = vector.times(1/sum);
        return vector.times(1/sum);
    }

    public void negative_sampling(String word) /*target word is the title + artist format string*/
    {
        double rand = new Random().nextDouble() / 10.0;
        int loop = G;
        while (loop != 0)
        {
            double cumulative = 0.0;
            for (String neg : vocab_neg.keySet())
            {
                cumulative += vocab_neg.get(neg);
                if (!negative_words.contains(neg)&&(cumulative >= rand)&&(!neg.contains(word)))
                {
                    negative_words.add(neg);
                    loop--;
                    break;
                }
            }
        }
    }

    public void update_hidden2output(String word, int column2)
    {
        negative_sampling(word);
        error = output_vector().minusEquals(vectorize(word));
        for (String neg : negative_words)
        {
            Matrix replace = new Matrix(N, 1);
            int column = vocab_table.get(neg);
            replace = hidden2output_weight.getMatrix(0, N-1, column, column);
            replace.minusEquals(hidden.times(error.get(column, 0)).times(L));
            hidden2output_weight.setMatrix(0, N-1, column, column, replace);
        }
        Matrix replace2 = new Matrix(N, 1);
        replace2 = hidden2output_weight.getMatrix(0, N-1, column2, column2);
        replace2.minusEquals(hidden.times(error.get(column2, 0)).times(L));
        hidden2output_weight.setMatrix(0, N-1, column2, column2, replace2);
    }

    public void update_input2hidden(String word)
    {
        int col = vocab_table.get(word);
        update_hidden2output(word, col);
        Matrix replace = new Matrix(N,1);
        replace = input2hidden_weight.getMatrix(col, col, 0, N-1).transpose();
        replace.minusEquals((hidden2output_weight.times(error).times(L/(double)C)));
        input2hidden_weight.setMatrix(col, col, 0, N-1, replace.transpose());
    }

    public void train_feed(int iter)
    {
        for (int count = 0; count < iter; count++)
        {
            factory = new IndoEuropeanTokenizerFactory();
            LowerCaseTokenizerFactory low = new LowerCaseTokenizerFactory(factory);
            EnglishStopTokenizerFactory stop = new EnglishStopTokenizerFactory(factory);
            PorterStemmerTokenizerFactory stem = new PorterStemmerTokenizerFactory(factory);

            for (String lyric_set : song_table.keySet())
            {
                Tokenization tokens = new Tokenization(lyric_set, factory);
                ArrayList<String> lyrics_words = new ArrayList<String>();
                for (String token : tokens.tokens())
                {
                    try {
                        String target = stem.modifyToken(stop.modifyToken(low.modifyToken(token)));
                        lyrics_words.add(target);
                    } catch (Exception e) { }
                }
                for (int i = 0; i < lyrics_words.size() - C + 1; i++)
                {
                    for (int j = i; j < C; j++)
                    {
                        context_words.add(lyrics_words.get(j));
                    }
                    update_input2hidden(song_table.get(lyric_set));
                    context_words.clear();
                    negative_words.clear();
                }
            }
            System.out.println("남은 횟수: " + (iter - count));
        }
    }

    public double cosine_sim(Matrix a, Matrix b)
    {
        double operand1 = 0;
        double operand2 = 0;
        double operand3 = 0;
        for (int i = 0; i < a.getRowDimension(); i++)
        {
            operand1 += a.get(i, 0) * a.get(i, 0);
            operand2 += b.get(i, 0) * b.get(i, 0);
            operand3 += a.get(i, 0) * b.get(i, 0);
        }
        return operand3 / (Math.sqrt(operand1) * Math.sqrt(operand2));
    }

    public void retrieve(String text)
    {
        factory = new IndoEuropeanTokenizerFactory();
        LowerCaseTokenizerFactory low = new LowerCaseTokenizerFactory(factory);
        EnglishStopTokenizerFactory stop = new EnglishStopTokenizerFactory(factory);
        PorterStemmerTokenizerFactory stem = new PorterStemmerTokenizerFactory(factory);

        String target = stem.modifyToken(stop.modifyToken(low.modifyToken(text)));
        context_words.add(target);
        Matrix result = new Matrix(V, 1);
        result = output_vector();
        for (String word : vocab_table.keySet())
        {
            System.out.print(String.format("Cosine Similarity: %20f\t", cosine_sim(result, vectorize(word))));
            System.out.println("V: " + V + " C: " + C + " N: " + N + " Learning Rate: " + L + " # of Negative Samples " + G + " " + word);
        }
    }


    public void song_crawl()
    {
        Document doc;
        String alpha = "abcdefghijklmnopqrstuvwxyz";
        char[] alpha_array = alpha.toCharArray();
        for (char letter : alpha_array)
        {
            try {
                doc = Jsoup.connect("http://www.azlyrics.com/" + letter + ".html").get();
                Elements list = doc.getElementsByClass("col-sm-6 text-center artist-col");
                for (Element sublist : list)
                {
                    for (Element item : sublist.select("a"))
                    {
                        String song_artist = item.text();
                        String link = item.attr("href");
                        try {
                            doc = Jsoup.connect("http://www.azlyrics.com/"+link).get();
                            Element albums = doc.getElementById("listAlbum");
                            for (Element album : albums.select("a"))
                            {

                                String link2 = album.attr("href");
                                link2 = link2.replace("..", "");
                                try {
                                    doc = Jsoup.connect("http://www.azlyrics.com"+link2).get();
                                    Elements contents = doc.getElementsByClass("col-xs-12 col-lg-8 text-center");
                                    Element lyrics = contents.select("div").get(7);
                                    Element title = contents.select("b").get(0);
                                    String song_title = title.text();
                                    String song_lyric = lyrics.text();
                                    song_lyrics.add(song_lyric);
                                    song_table.put(song_lyric, song_artist + " " + song_title);
                                    vocab_table.put(song_artist + " " + song_title, vocab_table.size());
                                    int i = 0;
                                } catch (Exception e) {}
                            }
                        } catch (Exception e) {}
                    }
                }
            } catch (Exception e) {}
        }
    }
}


