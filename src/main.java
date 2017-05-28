/**
 * Created by Maven Hyun on 2017-05-25.
 */
public class main {
    public static void main(String[] args) throws Exception {
        music2vec maven = new music2vec(6, 150, 0.001, 5); /* C, N , learning_rate*/
        maven.song_collect();
        maven.song_tokenize();
        maven.song_negative();
        maven.train_feed(1000);
        maven.retrieve("I have a good feeling that we could get lucky if we raise our cup to the stars");

        maven.song_crawl();

        return;
    }
}