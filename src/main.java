/**
 * Created by Maven Hyun on 2017-05-25.
 */
public class main {
    public static void main(String[] args) throws Exception {
        music2vec maven = new music2vec(8, 300, 0.001, 5); /* C, N , learning_rate*/
        maven.song_collect();
        maven.song_tokenize();
        maven.song_negative();
        maven.train_feed(200);
        maven.retrieve("lucky");

        /*maven.song_crawl();*/

        return;
    }
}