#include <exception>

class FileException : public std::exception{

	const char *msg;
public:

	
	const char *what() const throw() {return msg;}
	FileException() {msg="File operation error";}
	FileException(const char *m){msg=m;}
};
