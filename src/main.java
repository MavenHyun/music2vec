/**
 * Created by Maven Hyun on 2017-05-25.
 */
public class main {
    public static void main(String[] args) throws Exception {
        music2vec maven = new music2vec(4, 100, 0.001, 15); /* C, N , learning_rate*/
        maven.song_collect();
        maven.song_tokenize();
        maven.song_negative();
        maven.train_feed(5000);
        maven.retrieve("lucky");

        return;
    }
}