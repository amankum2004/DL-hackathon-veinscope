import '../entities/user.dart';
import '../repository/user_repository.dart';

class SignupUser {
  final UserRepository repository;

  SignupUser(this.repository);

  Future<User?> execute(User user) {
    return repository.signupUser(user);
  }
}
